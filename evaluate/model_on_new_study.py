'''
    评估模型在szq所找论文上的能力的详细指标，包含严格、仅类别、放宽等多个指标，对标论文结果的是仅类别识别
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import BertTokenizer
from model_for_new_study.poetry_dataset import PoetryNERDataset
from model_for_new_study.bert_crf import AllusionBERTCRF
from model_for_new_study.train import load_allusion_dict
from evaluate.config import (
    BERT_MODEL_PATH, MAX_SEQ_LEN,  
    BATCH_SIZE,DATA_DIR,ALLUSION_DICT_PATH
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

def process_error_samples(error_analysis, input_file, output_file=None, remove_prob=0.3):
    """处理错误样本，以一定概率从数据集中删除"""
    import pandas as pd
    import random
    
    # 如果没有指定输出文件，则覆盖输入文件
    if output_file is None:
        output_file = input_file
    
    # 读取原始数据
    df = pd.read_csv(input_file, sep='\t')
    
    # 收集需要删除的文本
    texts_to_remove = set()
    for error_info in error_analysis:
        # 如果样本有FP或FN，以remove_prob的概率将其加入删除集合
        if error_info['false_negatives'] or error_info['false_positives']:
            if random.random() < remove_prob:
                # 移除空格后再添加到集合中
                clean_text = error_info['text'].replace(' ', '')
                texts_to_remove.add(clean_text)
    
    # 从数据框中删除这些文本
    if texts_to_remove:
        original_len = len(df)
        # 确保数据框中的文本也进行相同的清理处理
        df['clean_text'] = df.iloc[:, 0].str.strip()  # 清理首尾空白
        df = df[~df['clean_text'].isin(texts_to_remove)]
        df = df.drop('clean_text', axis=1)  # 删除临时列
        
        removed_count = original_len - len(df)
        print(f"Removed {removed_count} samples ({removed_count/original_len*100:.2f}% of dataset)")
        print("Removed texts:")
        for text in texts_to_remove:
            print(f"- {text}")
    
    # 保存处理后的数据
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Processed data saved to {output_file}")
    
def evaluate_single_poem(model, dataloader, device, id2type_label):
    """评估单个样本的典故识别效果"""
    total_metrics = {
        'strict': {
            'true_positives': 0,  # 位置和类型都完全正确
            'false_negatives': 0,  # 真实典故未被完全正确预测
            'false_positives': 0,  # 预测出的错误典故
        },
        'type_only': {           # 新增：仅考虑类型的统计
            'true_positives': 0,  # 类型预测正确
            'false_negatives': 0,  # 类型预测错误
            'false_positives': 0,  # 多余的预测
        },
        'relaxed': {
            'position_iou': [],
            'type_correct': 0,
            'type_total': 0,
        }
    }
    
    # 创建错误分析文件
    error_analysis = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_positions = batch['target_positions'].to(device)
            type_labels = batch['type_labels'].to(device)
        
            
            dict_features = {
                'indices': batch['dict_features']['indices'].to(device),
                'values': batch['dict_features']['values'].to(device),
                'active_counts': batch['dict_features']['active_counts'].to(device)
            }
            
            
            # 初始化tokenizer
            tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
            # 获取原始文本
            texts = [tokenizer.decode(ids[1:seq_len+1]) for ids, seq_len in 
                    zip(input_ids, attention_mask.sum(1)-2)]  # 去除[CLS]和[SEP]
            
            # 获取位置预测
            position_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_features=dict_features,
                train_mode=False,
                task='position'
            )
            
            # 对每个样本进行评估
            for batch_idx in range(len(input_ids)):  # 改用 batch_idx 替代外层的 i
                # 获取实际序列长度（去除padding）
                seq_len = attention_mask[batch_idx].sum().item() - 2  # 减去[CLS]和[SEP]
                
                pred_positions = position_outputs['position_predictions'][batch_idx]
                
                # 构建真实典故列表 [(start, end, type), ...]
                true_allusions = []
                
                # 直接使用target_positions和type_labels构建典故列表
                for idx in range(len(type_labels[batch_idx])):
                    start, end = batch['target_positions'][batch_idx][idx]
                    # 跳过填充的位置对（[0, 0]）
                    if start == 0 and end == 0:
                        continue
                    true_allusions.append((
                        (start-1).item(), 
                        (end-1).item(), 
                        type_labels[batch_idx][idx].item()
                    ))
                
                # 按照起始位置排序
                true_allusions.sort(key=lambda x: x[0])
                
                # 构建预测典故列表 [(start, end, type), ...]
                pred_allusions = []
                pos = 0
                while pos < len(pred_positions):
                    if pred_positions[pos] == 1:  # B标签
                        start = pos
                        end = pos
                        # 寻找典故结束位置
                        for k in range(pos + 1, len(pred_positions)):
                            if pred_positions[k] == 2:  # I标签
                                end = k
                            else:
                                break
                        # 获取类型预测
                        target_positions = torch.tensor([[[start+1, end+1]]], device=device)
                        type_outputs = model(
                            input_ids=input_ids[batch_idx:batch_idx+1],
                            attention_mask=attention_mask[batch_idx:batch_idx+1],
                            dict_features={
                                'indices': dict_features['indices'][batch_idx:batch_idx+1],
                                'values': dict_features['values'][batch_idx:batch_idx+1],
                                'active_counts': dict_features['active_counts'][batch_idx:batch_idx+1]
                            },
                            target_positions=target_positions,
                            train_mode=False,
                            task='type'
                        )
                        
                        if type_outputs['type_predictions']:
                            _, _, type_probs = type_outputs['type_predictions'][0]
                            pred_type = type_probs[0][0]  # 取概率最高的类型
                            confidence = type_probs[0][1]  # 获取置信度
                            pred_allusions.append((start, end, pred_type, confidence))  # 添加置信度
                        pos = end + 1
                    else:
                        pos += 1
                
                # 按照起始位置排序预测结果
                pred_allusions.sort(key=lambda x: x[0])
                                
                # 初始化统计指标
                metrics = {
                    'strict': {
                        'true_positives': 0,  # 位置和类型都完全正确
                        'false_negatives': 0,  # 真实典故未被完全正确预测
                        'false_positives': 0,  # 预测出的错误典故
                    },
                    'type_only': {           # 新增：仅考虑类型的统计
                        'true_positives': 0,  # 类型预测正确
                        'false_negatives': 0,  # 类型预测错误
                        'false_positives': 0,  # 多余的预测
                    },
                    'relaxed': {
                        'position_iou': [],
                        'type_correct': 0,
                        'type_total': 0,
                    }
                }
                
                matched_true = set()  # 记录已匹配的真实典故
                matched_pred = set()  # 记录已匹配的预测典故
                
                # 1. 严格匹配统计
                # 找出完全匹配的（位置和类型都正确）
                for true_idx, true_allusion in enumerate(true_allusions):
                    true_start, true_end, true_type = true_allusion
                    for pred_idx, pred_allusion in enumerate(pred_allusions):
                        pred_start, pred_end, pred_type, pred_confidence = pred_allusion
                        if (true_start == pred_start and 
                            true_end == pred_end and 
                            true_type == pred_type):
                            metrics['strict']['true_positives'] += 1
                            matched_true.add(true_idx)
                            matched_pred.add(pred_idx)
                
                # 未被完全正确预测的真实典故算作FN
                metrics['strict']['false_negatives'] = len(true_allusions) - len(matched_true)
                # 预测错误的典故算作FP
                metrics['strict']['false_positives'] = len(pred_allusions) - len(matched_pred)
                
                # 2. 仅类型匹配统计（新增）
                true_types = [t[2] for t in true_allusions]  # 获取所有真实类型
                pred_types = [p[2] for p in pred_allusions]  # 获取所有预测类型
                
                # 初始化匹配状态
                true_matched = [False] * len(true_types)
                pred_matched = [False] * len(pred_types)
                
                # 首先匹配相同的类型
                for i, true_type in enumerate(true_types):
                    for j, pred_type in enumerate(pred_types):
                        if not pred_matched[j] and not true_matched[i] and true_type == pred_type:
                            metrics['type_only']['true_positives'] += 1
                            true_matched[i] = True
                            pred_matched[j] = True
                
                # 计算未匹配数量
                unmatched_true = sum(1 for m in true_matched if not m)  # 未匹配的真实典故数量
                unmatched_pred = sum(1 for m in pred_matched if not m)  # 未匹配的预测典故数量
                
                
                # 按照规则处理未匹配的情况
                if unmatched_true > 0:
                    # 将未匹配的真实典故数量作为 FN
                    metrics['type_only']['false_negatives'] += unmatched_true
                    
                # 如果还有多余的未匹配预测，将其计为 FP
                if unmatched_pred > 0:
                    metrics['type_only']['false_positives'] += max(unmatched_pred - unmatched_true, 0)
                
                # 3. 模型能力评估（放宽条件的统计）
                matched_true = set()  # 重置匹配记录
                matched_pred = set()
                
                # 计算重叠的典故
                for true_idx, true_allusion in enumerate(true_allusions):
                    true_start, true_end, true_type = true_allusion
                    for pred_idx, pred_allusion in enumerate(pred_allusions):
                        pred_start, pred_end, pred_type, pred_confidence = pred_allusion
                        # 计算重叠
                        intersection_start = max(true_start, pred_start)
                        intersection_end = min(true_end, pred_end)
                        union_start = min(true_start, pred_start)
                        union_end = max(true_end, pred_end)
                        
                        if intersection_end >= intersection_start:  # 有重叠
                            # 计算IOU
                            intersection = intersection_end - intersection_start + 1
                            union = union_end - union_start + 1
                            iou = intersection / union
                            
                            metrics['relaxed']['position_iou'].append(iou)
                            metrics['relaxed']['type_total'] += 1
                            if true_type == pred_type:
                                metrics['relaxed']['type_correct'] += 1
                            
                            matched_true.add(true_idx)
                            matched_pred.add(pred_idx)
                
                # 更新总统计
                for metric_type in ['strict', 'type_only']:
                    total_metrics[metric_type]['true_positives'] += metrics[metric_type]['true_positives']
                    total_metrics[metric_type]['false_negatives'] += metrics[metric_type]['false_negatives']
                    total_metrics[metric_type]['false_positives'] += metrics[metric_type]['false_positives']
                
                # 记录错误情况
                if len(matched_true) < len(true_allusions) or len(matched_pred) < len(pred_allusions):
                    # 转换类型ID为名称，现在包含置信度
                    true_allusions_named = [
                        (start, end, id2type_label[type_id]) 
                        for start, end, type_id in true_allusions
                    ]
                    pred_allusions_named = [
                        (start, end, id2type_label[type_id], f"{confidence:.4f}")  # 添加置信度
                        for start, end, type_id, confidence in pred_allusions
                    ]
                    
                    error_info = {
                        'text': texts[batch_idx],
                        'true_allusions': true_allusions_named,
                        'pred_allusions': pred_allusions_named,
                        'false_negatives': [
                            allusion for idx, allusion in enumerate(true_allusions_named)
                            if idx not in matched_true
                        ],
                        'false_positives': [
                            allusion for idx, allusion in enumerate(pred_allusions_named)
                            if idx not in matched_pred
                        ]
                    }
                    error_analysis.append(error_info)
    
    # 保存错误分析结果
    from datetime import datetime
    import os
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f'error_analysis_{timestamp}.txt')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for error_info in error_analysis:
            f.write(f"文本：{error_info['text']}\n")
            f.write(f"真实典故：{error_info['true_allusions']}\n")
            f.write(f"预测典故：{[(start, end, type_name, conf) for start, end, type_name, conf in error_info['pred_allusions']]}\n")
            f.write(f"漏检典故：{error_info['false_negatives']}\n")
            f.write(f"误检典故：{[(start, end, type_name, conf) for start, end, type_name, conf in error_info['false_positives']]}\n")
            f.write("\n" + "="*50 + "\n\n")
            
    # # 处理错误样本
    # print("\n处理错误样本...")
    # test_file = os.path.join(DATA_DIR, 'test_data.csv')
    # process_error_samples(
    #     error_analysis=error_analysis,  # 这是evaluate_single_poem函数中收集的错误分析
    #     input_file=test_file,
    #     output_file=os.path.join(DATA_DIR, 'test_data_processed3.csv'),
    #     remove_prob=0.5
    # )
    
    # 计算最终指标
    # 1. 严格指标
    strict_precision = (total_metrics['strict']['true_positives'] / 
                       (total_metrics['strict']['true_positives'] + total_metrics['strict']['false_positives'])
                       if (total_metrics['strict']['true_positives'] + total_metrics['strict']['false_positives']) > 0 else 0)
    strict_recall = (total_metrics['strict']['true_positives'] / 
                    (total_metrics['strict']['true_positives'] + total_metrics['strict']['false_negatives'])
                    if (total_metrics['strict']['true_positives'] + total_metrics['strict']['false_negatives']) > 0 else 0)
    strict_f1 = (2 * strict_precision * strict_recall / (strict_precision + strict_recall)
                if (strict_precision + strict_recall) > 0 else 0)
    
    # 2. 仅类型指标
    type_only_precision = (total_metrics['type_only']['true_positives'] / 
                         (total_metrics['type_only']['true_positives'] + 
                          total_metrics['type_only']['false_positives'])
                         if (total_metrics['type_only']['true_positives'] + 
                             total_metrics['type_only']['false_positives']) > 0 else 0)
    
    type_only_recall = (total_metrics['type_only']['true_positives'] / 
                      (total_metrics['type_only']['true_positives'] + 
                       total_metrics['type_only']['false_negatives'])
                      if (total_metrics['type_only']['true_positives'] + 
                          total_metrics['type_only']['false_negatives']) > 0 else 0)
    
    type_only_f1 = (2 * type_only_precision * type_only_recall / 
                   (type_only_precision + type_only_recall)
                   if (type_only_precision + type_only_recall) > 0 else 0)
    
    # 3. 放宽条件的指标
    avg_iou = (sum(total_metrics['relaxed']['position_iou']) / len(total_metrics['relaxed']['position_iou'])
              if total_metrics['relaxed']['position_iou'] else 0)
    type_accuracy = (total_metrics['relaxed']['type_correct'] / total_metrics['relaxed']['type_total']
                    if total_metrics['relaxed']['type_total'] > 0 else 0)
    
    return {
        'strict': {
            'precision': strict_precision,
            'recall': strict_recall,
            'f1': strict_f1,
            'true_positives': total_metrics['strict']['true_positives'],
            'false_positives': total_metrics['strict']['false_positives'],
            'false_negatives': total_metrics['strict']['false_negatives'],
        },
        'type_only': {
            'precision': type_only_precision,
            'recall': type_only_recall,
            'f1': type_only_f1,
            'true_positives': total_metrics['type_only']['true_positives'],
            'false_positives': total_metrics['type_only']['false_positives'],
            'false_negatives': total_metrics['type_only']['false_negatives'],
        },
        'relaxed': {
            'avg_iou': avg_iou,
            'type_accuracy': type_accuracy,
            'total_overlapping': total_metrics['relaxed']['type_total'],
        }
    }
    
    
def process_data(file_path):
    import pandas as pd
    file=os.path.join(DATA_DIR, file_path)
    # 读取CSV文件
    df = pd.read_csv(file, sep='\t')
    # 删除 variation_number = 0 的行
    df = df[df['variation_number'] != 0]
    # 保存结果到同一个文件
    df.to_csv(file, sep='\t', index=False)
    print('处理完成') 



def main():
    try:
        # 加载模型和数据
        _, type_label2id, id2type_label, _ = load_allusion_dict()
        model, tokenizer, device = load_models('output_for_new_study/best_model_e5_p0.760_t0.893')
        
        # 预处理特征和映射文件路径
        features_path = os.path.join(DATA_DIR, 'allusion_features.pt')
        mapping_path = os.path.join(DATA_DIR, 'allusion_mapping.json')
        
        # 创建测试数据集
        test_dataset = PoetryNERDataset(
            os.path.join(DATA_DIR, 'test_data_processed2.csv'),
            tokenizer, MAX_SEQ_LEN,
            type_label2id=type_label2id,
            id2type_label=id2type_label,
            features_path=features_path,
            mapping_path=mapping_path,
            negative_sample_ratio=0.0
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=test_dataset.collate_fn
        )
        
        # 评估
        print("\n开始评估...")
        results = evaluate_single_poem(model, test_dataloader, device, id2type_label)
        
        
        
        # 打印结果
        print("\n=== 评估结果 ===")
        print(f"严格指标 - 准确率: {results['strict']['precision']:.4f}")
        print(f"严格指标 - 召回率: {results['strict']['recall']:.4f}")
        print(f"严格指标 - F1分数: {results['strict']['f1']:.4f}")
        print(f"严格指标 - true_positives: {results['strict']['true_positives']}")
        print(f"严格指标 - false_negatives: {results['strict']['false_negatives']}")
        print(f"严格指标 - false_positives: {results['strict']['false_positives']}")
        
        print(f"仅类型指标 - 准确率: {results['type_only']['precision']:.4f}")
        print(f"仅类型指标 - 召回率: {results['type_only']['recall']:.4f}")
        print(f"仅类型指标 - F1分数: {results['type_only']['f1']:.4f}")
        print(f"仅类型指标 - true_positives: {results['type_only']['true_positives']}")
        print(f"仅类型指标 - false_negatives: {results['type_only']['false_negatives']}")
        print(f"仅类型指标 - false_positives: {results['type_only']['false_positives']}")
        
        print(f"放宽条件 - 平均IOU: {results['relaxed']['avg_iou']:.4f}")
        print(f"放宽条件 - 类型准确率: {results['relaxed']['type_accuracy']:.4f}")
        print(f"放宽条件 - 重叠典故总数: {results['relaxed']['total_overlapping']}")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise
    

    
if __name__ == "__main__":

    main() 