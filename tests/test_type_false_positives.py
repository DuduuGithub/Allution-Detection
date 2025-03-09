import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import BertTokenizer
from model.poetry_dataset import PoetryNERDataset
from model.bert_crf import AllusionBERTCRF
from model.train import load_allusion_dict
from model.config import (
    BERT_MODEL_PATH, MAX_SEQ_LEN, SAVE_DIR, 
    DATA_DIR, BATCH_SIZE
)
from torch.utils.data import DataLoader
import pandas as pd
import random
from collections import defaultdict
from sklearn.metrics import classification_report

def test_type_false_positives():
    """测试类型识别模型对非典故位置的误判情况"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    allusion_dict, type_label2id, id2type_label, num_types = load_allusion_dict()
    dict_size = len(allusion_dict)
    
    # 创建并加载模型
    model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size).to(device)
    checkpoint = torch.load(f'{SAVE_DIR}/best_model_type.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 预处理特征和映射文件路径
    features_path = os.path.join(DATA_DIR, 'allusion_features_strictly_dict.pt')
    mapping_path = os.path.join(DATA_DIR, 'allusion_mapping_strictly_dict.json')
    
    # 创建数据集
    test_dataset = PoetryNERDataset(
        file_path=os.path.join(DATA_DIR, '4_test_type.csv'),#其实用不到
        tokenizer=tokenizer,
        max_len=MAX_SEQ_LEN,
        type_label2id=type_label2id,
        id2type_label=id2type_label,
        task='type',
        features_path=features_path,
        mapping_path=mapping_path,
        positive_sample_ratio=0.0
    )
    
    print(f"加载的数据集样本数: {len(test_dataset)}")
    
    # 创建dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    
    # 统计变量
    total_predictions = 0
    false_positive_count = 0
    type_confusion_matrix = defaultdict(int)
    
    # 进行预测
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            dict_features = batch['dict_features']
            target_positions = batch['target_positions'].to(device)
            
            if dict_features is not None:
                dict_features = {k: v.to(device) for k, v in dict_features.items()}
            
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_features=dict_features,
                task='type',
                target_positions=target_positions
            )
            
            predictions = output['predictions'][:, 0]  # 取top1预测
            
            # 统计结果
            batch_size = len(predictions)
            if total_predictions + batch_size > 150:
                # 只取需要的数量
                predictions = predictions[:150 - total_predictions]
            
            total_predictions += len(predictions)
            for pred in predictions:
                if pred != 0:  # 非0表示预测为某种典故类型
                    false_positive_count += 1
                    pred_type_name = id2type_label[pred.item()]
                    type_confusion_matrix[pred_type_name] += 1
                    
            if total_predictions >= 150:
                break
    
    # 计算误报率
    false_positive_rate = false_positive_count / total_predictions if total_predictions > 0 else 0
    
    # 输出统计结果
    print("\n=== 非典故位置预测统计 ===")
    print(f"总预测次数: {total_predictions}")
    print(f"误报数量: {false_positive_count}")
    print(f"误报率: {false_positive_rate:.4f}")
    
    print("\n=== 误报类型分布 ===")
    sorted_confusion = sorted(type_confusion_matrix.items(), key=lambda x: x[1], reverse=True)
    for type_name, count in sorted_confusion:
        print(f"{type_name}: {count} ({count/false_positive_count*100:.2f}%)")

if __name__ == "__main__":
    test_type_false_positives() 