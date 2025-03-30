import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import BertTokenizer
from model_for_new_study.poetry_dataset import PoetryNERDataset
from model_for_new_study.bert_crf import AllusionBERTCRF,prepare_sparse_features
from model_for_new_study.train import load_allusion_dict
from evaluate.config import (
    BERT_MODEL_PATH, MAX_SEQ_LEN,  
    BATCH_SIZE, DATA_DIR, ALLUSION_DICT_PATH
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

def load_models(model_name):
    """加载预训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    # 加载典故词典以获取类型数量
    allusion_dict, _, _, num_types = load_allusion_dict(ALLUSION_DICT_PATH)
    dict_size = len(allusion_dict)
    
    # 创建一个模型实例
    model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size=dict_size).to(device)
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAVE_DIR = os.path.join(PROJECT_ROOT, 'trained_result')
    
    # 加载模型参数并处理多余的键
    position_checkpoint = torch.load(f'{SAVE_DIR}/{model_name}.pt', map_location=device)
    
    print('testing model path:', f'{SAVE_DIR}/{model_name}.pt')
    
    state_dict = position_checkpoint['model_state_dict']
    
    # 移除多余的键
    for key in list(state_dict.keys()):
        if key not in model.state_dict():
            print(f"Removing unexpected key: {key}")
            del state_dict[key]
    
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, device

def prepare_batch_data(texts, tokenizer, allusion_dict, features_path=None, mapping_path=None):
    """准备批量数据，优先使用预处理特征，找不到时才动态生成"""
    # 尝试加载预处理的特征和映射
    precomputed_features = None
    sentence_to_id = None
    if features_path and mapping_path and os.path.exists(features_path) and os.path.exists(mapping_path):
        try:
            precomputed_features = torch.load(features_path)
            import json
            with open(mapping_path, 'r', encoding='utf-8') as f:
                sentence_to_id = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load precomputed features: {e}")
    
    # 获取最大长度
    max_text_len = max(len(text) for text in texts)
    
    # 准备batch数据的列表
    batch_texts = []
    batch_input_ids = []
    batch_attention_mask = []
    indices_list = []
    values_list = []
    active_counts_list = []
    
    # 处理每个文本
    for text in texts:
        batch_texts.append(text)
        
        # BERT tokenization
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_text_len + 2,  # +2 for [CLS] and [SEP]
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 获取input_ids和attention_mask
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        
        # 尝试使用预处理特征
        if precomputed_features is not None and sentence_to_id is not None and text in sentence_to_id:
            sent_id = sentence_to_id[text]
            indices = precomputed_features['indices'][sent_id]
            values = precomputed_features['values'][sent_id]
            active_counts = precomputed_features['active_counts'][sent_id]
        else:
            # 如果找不到预处理特征，则动态生成
            text_features = prepare_sparse_features([text], allusion_dict)
            indices = text_features['indices'].squeeze(0)
            values = text_features['values'].squeeze(0)
            active_counts = text_features['active_counts'].squeeze(0)
        
        # 获取当前序列长度
        seq_len = len(input_ids)
        
        # 处理字典特征的维度
        indices = indices[:seq_len]
        values = values[:seq_len]
        active_counts = active_counts[:seq_len]
        
        # 补全到最大长度
        if indices.size(0) < seq_len:
            pad_len = seq_len - indices.size(0)
            indices = torch.cat([indices, torch.zeros((pad_len, 5), dtype=torch.long)], dim=0)
            values = torch.cat([values, torch.zeros((pad_len, 5), dtype=torch.float)], dim=0)
            active_counts = torch.cat([active_counts, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        
        indices_list.append(indices)
        values_list.append(values)
        active_counts_list.append(active_counts)
    
    # 堆叠所有张量
    return {
        'text': batch_texts,
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
        'dict_features': {
            'indices': torch.stack(indices_list),
            'values': torch.stack(values_list),
            'active_counts': torch.stack(active_counts_list)
        }
    }

def evaluate_single_poem(model, data, tokenizer, device, allusion_dict, id2type_label, features_path=None, mapping_path=None):
    """评估单个样本的用典判断效果"""
    metrics = {
        'true_positives': 0,
        'false_positives': 0,
        'true_negatives': 0,
        'false_negatives': 0,
    }
    
    error_analysis = []
    
    # 按批次处理数据
    for i in tqdm(range(0, len(data), BATCH_SIZE)):
        batch_data = data[i:i + BATCH_SIZE]
        texts = [item[0] for item in batch_data]
        variation_numbers = [item[2] for item in batch_data]  # 获取variation_number
        
        # 准备批量数据
        batch = prepare_batch_data(texts, tokenizer, allusion_dict, features_path, mapping_path)
        
        # 将数据移到设备上
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        dict_features = {k: v.to(device) for k, v in batch['dict_features'].items()}
        
        # 获取位置预测
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dict_features=dict_features,
            train_mode=False,
            task='position'
        )
        
        pred_positions = outputs['position_predictions']
        
        for batch_idx in range(len(texts)):
            # 通过variation_number判断是否有典故
            has_allusion_true = (variation_numbers[batch_idx] != 0)
            
            # 获取预测结果
            has_allusion_pred = any(pos in [1, 2] for pos in pred_positions[batch_idx])
            
            # 更新指标
            if has_allusion_true and has_allusion_pred:
                metrics['true_positives'] += 1
            elif has_allusion_true and not has_allusion_pred:
                metrics['false_negatives'] += 1
            elif not has_allusion_true and has_allusion_pred:
                metrics['false_positives'] += 1
            else:
                metrics['true_negatives'] += 1
            
            # 记录错误案例
            if (has_allusion_true != has_allusion_pred):
                error_info = {
                    'text': texts[batch_idx],
                    'true_label': '有典故' if has_allusion_true else '无典故',
                    'pred_label': '有典故' if has_allusion_pred else '无典故',
                    'error_type': 'FN' if has_allusion_true else 'FP'
                }
                error_analysis.append(error_info)
    
    # 计算评估指标
    total = sum(metrics.values())
    precision = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0
    recall = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives']) if (metrics['true_positives'] + metrics['false_negatives']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (metrics['true_positives'] + metrics['true_negatives']) / total if total > 0 else 0
    
    return {
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'true_positives': metrics['true_positives'],
            'false_positives': metrics['false_positives'],
            'true_negatives': metrics['true_negatives'],
            'false_negatives': metrics['false_negatives'],
        },
        'error_analysis': error_analysis
    }

def main():
    # 加载模型和数据
    allusion_dict, type_label2id, id2type_label, _ = load_allusion_dict(ALLUSION_DICT_PATH)
    model, tokenizer, device = load_models('output_for_new_study/best_model_e5_p0.760_t0.893')
    
    # 预处理特征和映射文件路径
    features_path = os.path.join(DATA_DIR, 'allusion_features.pt')
    mapping_path = os.path.join(DATA_DIR, 'allusion_mapping.json')
    
    # 读取数据
    import pandas as pd
    data_path = os.path.join(DATA_DIR, 'test_data_for_task_1.csv')
    df = pd.read_csv(data_path, sep='\t')
    data = list(df.itertuples(index=False, name=None))
    
    # 评估
    print("\n开始评估...")
    results = evaluate_single_poem(model, data, tokenizer, device, allusion_dict, id2type_label, 
                                 features_path=features_path, mapping_path=mapping_path)
    
    # 打印结果
    print("\n=== 用典判断评估结果 ===")
    print(f"准确率: {results['metrics']['precision']:.4f}")
    print(f"召回率: {results['metrics']['recall']:.4f}")
    print(f"F1分数: {results['metrics']['f1']:.4f}")
    print(f"整体准确率: {results['metrics']['accuracy']:.4f}")
    print(f"\n详细统计:")
    print(f"正确预测有典故 (TP): {results['metrics']['true_positives']}")
    print(f"错误预测有典故 (FP): {results['metrics']['false_positives']}")
    print(f"正确预测无典故 (TN): {results['metrics']['true_negatives']}")
    print(f"错误预测无典故 (FN): {results['metrics']['false_negatives']}")
    
    # 保存错误分析
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_path = os.path.join(os.path.dirname(__file__), f'error_analysis_task1_{timestamp}.txt')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=== 错误案例分析 ===\n\n")
        for error in results['error_analysis']:
            f.write(f"文本：{error['text']}\n")
            f.write(f"真实标签：{error['true_label']}\n")
            f.write(f"预测标签：{error['pred_label']}\n")
            f.write(f"错误类型：{error['error_type']}\n")
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"\n错误分析已保存到: {save_path}")

if __name__ == "__main__":
    main() 