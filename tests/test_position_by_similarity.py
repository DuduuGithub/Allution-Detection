import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from model.config import DATA_DIR
from process.process_dict_features import load_sentence_mappings, load_features
from sklearn.metrics import classification_report

def predict_positions_by_similarity(features, similarity_threshold=0.5):
    """
    根据相似度预测典故位置
    Args:
        features: 特征字典，包含 active_counts, indices, values
        similarity_threshold: 相似度阈值
    Returns:
        predictions: 位置预测序列 [0, 0, 1, 1, 1, 0, ...]  (0:O, 1:I)
    """
    seq_len = len(features['active_counts'])
    predictions = np.zeros(seq_len, dtype=int)
    
    # 遍历每个位置
    i = 0
    while i < seq_len:
        active_count = features['active_counts'][i].item()
        if active_count == 0:
            i += 1
            continue
            
        # 检查该位置的最大相似度
        max_similarity = max(features['values'][i][:active_count].numpy())
        
        if max_similarity >= similarity_threshold:
            # 标记为典故位置
            predictions[i] = 1
            
            # 向后查找连续的高相似度位置
            j = i + 1
            while j < seq_len:
                next_active_count = features['active_counts'][j].item()
                if next_active_count > 0:
                    next_max_similarity = max(features['values'][j][:next_active_count].numpy())
                    if next_max_similarity >= similarity_threshold:
                        predictions[j] = 1  # 标记为典故位置
                        j += 1
                        continue
                break
            i = j
        else:
            i += 1
    
    return predictions

def test_position_prediction():
    """测试基于相似度的位置预测"""
    # 加载特征和映射
    mapping_path = os.path.join(DATA_DIR, 'allusion_mapping_strictly_dict.json')
    features_path = os.path.join(DATA_DIR, 'allusion_features_strictly_dict.pt')
    
    sentence_to_id, _ = load_sentence_mappings(mapping_path)
    all_features = load_features(features_path)
    
    # 加载验证数据
    val_file = os.path.join(DATA_DIR, '4_val_position_no_bug.csv')
    val_data = pd.read_csv(val_file, sep='\t')
    
    all_true_labels = []
    all_pred_labels = []
    
    # 处理每个样本
    for _, row in val_data.iterrows():
        sentence = row['sentence']
        if sentence not in sentence_to_id:
            continue
            
        # 获取特征
        sent_id = sentence_to_id[sentence]
        features = all_features[sent_id]
        
        # 生成真实标签序列
        seq_len = len(features['active_counts'])
        true_labels = [0] * seq_len  # 初始化为全0（非典故）
        
        # 解析 transformed_allusion 列
        if pd.notna(row['transformed_allusion']):
            allusions = row['transformed_allusion'].strip('[]').split('];[')
            for allusion in allusions:
                if not allusion:
                    continue
                # 清理字符串并解析位置
                parts = allusion.strip().split(',')
                try:
                    # 提取位置信息（去除可能的空格和方括号）
                    positions = [int(pos.strip(' []')) for pos in parts[:-1]]  # 最后一个元素是典故名称
                    if positions:
                        # 标记所有典故位置为1
                        for pos in positions:
                            true_labels[pos] = 1
                except ValueError as e:
                    print(f"Warning: 解析位置出错: {allusion}")
                    continue
        
        # 预测位置
        pred_labels = predict_positions_by_similarity(features, similarity_threshold=0.5)
        
        # 记录结果
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
    
    # 计算指标
    print("\n=== 位置预测评估报告 ===")
    print(classification_report(all_true_labels, all_pred_labels, digits=4))
    
    # 计算详细指标
    correct = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == p)
    total = len(all_true_labels)
    
    print("\n=== 详细统计 ===")
    print(f"总体准确率: {correct/total:.4f} ({correct}/{total})")
    
    # 计算每个标签的准确率
    for label, name in [(0, 'O'), (1, 'I')]:
        label_positions = [i for i, t in enumerate(all_true_labels) if t == label]
        if label_positions:
            correct_label = sum(1 for i in label_positions if all_pred_labels[i] == label)
            print(f"标签 {name} 准确率: {correct_label/len(label_positions):.4f} "
                  f"({correct_label}/{len(label_positions)})")

if __name__ == "__main__":
    test_position_prediction() 