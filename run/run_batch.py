import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import BertTokenizer
from model.bert_crf import AllusionBERTCRF, prepare_sparse_features
from model.train import load_allusion_dict
from model.config import (
    BERT_MODEL_PATH, MAX_SEQ_LEN,
    BATCH_SIZE, DATA_DIR
)

def load_model(model_name):
    """加载预训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    # 加载典故词典以获取类型数量
    allusion_dict, _, id2type_label, num_types = load_allusion_dict()
    dict_size = len(allusion_dict)
    
    # 创建模型实例
    model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size=dict_size).to(device)
    
    # 加载模型参数
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAVE_DIR = os.path.join(PROJECT_ROOT, 'trained_result')
    
    checkpoint = torch.load(f'{SAVE_DIR}/{model_name}.pt', map_location=device)
    
    # 处理模型状态字典
    state_dict = checkpoint['model_state_dict']
    for key in list(state_dict.keys()):
        if key not in model.state_dict():
            print(f"Removing unexpected key: {key}")
            del state_dict[key]
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, tokenizer, device, allusion_dict, id2type_label

def prepare_batch_data(texts, tokenizer, allusion_dict):
    """
    准备批量数据，参考 PoetryNERDataset 的 collate_fn 函数
    """
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
        
        # 处理字典特征
        text_features = prepare_sparse_features([text], allusion_dict)
        
        # 获取当前序列长度
        seq_len = len(input_ids)
        
        # 处理字典特征的维度
        indices = text_features['indices'].squeeze(0)[:seq_len]
        values = text_features['values'].squeeze(0)[:seq_len]
        active_counts = text_features['active_counts'].squeeze(0)[:seq_len]
        
        # 补全到最大长度（保持[CLS]位置为0）
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
        'texts': batch_texts,
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
        'dict_features': {
            'indices': torch.stack(indices_list),
            'values': torch.stack(values_list),
            'active_counts': torch.stack(active_counts_list)
        }
    }

def predict_batch(texts, model, tokenizer, device, allusion_dict, id2type_label):
    """批量预测多个文本中的典故"""
    # 准备批量数据
    batch_data = prepare_batch_data(texts, tokenizer, allusion_dict)
    # 将数据移到设备上
    input_ids = batch_data['input_ids'].to(device)
    attention_mask = batch_data['attention_mask'].to(device)
    dict_features = {
        'indices': batch_data['dict_features']['indices'].to(device),
        'values': batch_data['dict_features']['values'].to(device),
        'active_counts': batch_data['dict_features']['active_counts'].to(device)
    }
    
    batch_results = []
    
    with torch.no_grad():
        # 位置识别
        position_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dict_features=dict_features,
            train_mode=False,
            task='position'
        )
        
        position_predictions = position_outputs['position_predictions']
        print('position_predictions:', position_predictions)
        # 处理每个样本
        for batch_idx, (text, position_pred) in enumerate(zip(texts, position_predictions)):
            text_results = []
            
            i = 0
            while i < len(position_pred):
                if position_pred[i] == 1:  # B标签
                    start = i
                    end = i
                    # 寻找典故结束位置
                    for j in range(i + 1, len(position_pred)):
                        if position_pred[j] == 2:  # I标签
                            end = j
                        else:
                            break
                    
                    # 准备类型识别的输入
                    target_positions = torch.tensor([[[start+1, end+1]]], device=device)
                    
                    # 类型识别
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
                    
                    # 处理类型预测结果
                    type_predictions = type_outputs['type_predictions']
                    if type_predictions:
                        _, _, type_probs = type_predictions[0]
                        text_results.append((start, end, type_probs)) 
                        print('text_results:', text_results)
                    i = end
                else:
                    i += 1
            
            batch_results.append(text_results)
    
    return batch_results

def main():
    # 加载模型
    model, tokenizer, device, allusion_dict, id2type_label = load_model('output_jointly_train_normalize_loss/best_model_3.17.22.09')
    
    # 测试用例
    test_poems = [
        "桃源避秦人不见，武陵渔父独知处。",
        "一入石渠署，三闻宫树蝉。",
        "穷途行泣玉，愤路未藏金。",
        "莱子多嘉庆，陶公得此生。",
        "扬风非赠扇，易俗是张琴。"
    ]
    
    # 批量预测
    results = predict_batch(test_poems, model, tokenizer, device, allusion_dict, id2type_label)
    
    # 打印结果
    for text, text_results in zip(test_poems, results):
        print(f"\n诗句: {text}")
        print("预测结果:")
        for start, end, predictions in text_results:
            allusion_text = text[start:end+1]
            print(f"\n典故: {allusion_text} ({start}-{end})")
            for pred_type, prob in predictions:
                type_name = id2type_label[pred_type]
                print(f"  - {type_name}: {prob:.3f}")

if __name__ == "__main__":
    main() 