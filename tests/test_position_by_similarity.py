'''
    使用字典特征进行位置预测的效果监测。
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from model.config import DATA_DIR
from model.train import load_allusion_dict
from process.process_dict_features import load_sentence_mappings, load_features
from sklearn.metrics import classification_report

def predict_positions_and_types_by_similarity(features, type2id,id2type, similarity_threshold=0.5):
    """
    根据相似度预测典故位置和类型
    Args:
        features: 特征字典，包含 active_counts, indices, values
        type2id: 典故类型到ID的映射字典
        similarity_threshold: 相似度阈值
    Returns:
        position_predictions: 位置预测序列 ['O', 'B-1', 'I-1', 'O', ...] 
        type_predictions: 类型预测序列 [0, 1, 1, 0, ...] 
    """
    seq_len = len(features['active_counts'])
    position_predictions = ['O'] * seq_len
    type_predictions = [0] * seq_len
    
    i = 0
    while i < seq_len:
        active_count = features['active_counts'][i].item()
        if active_count == 0:
            i += 1
            continue
            
        # 获取该位置的最大相似度及其对应的典故名称
        values = features['values'][i][:active_count].numpy()
        indices = features['indices'][i][:active_count].numpy()
        max_similarity_idx = np.argmax(values)
        max_similarity = values[max_similarity_idx]
        allusion_id = indices[max_similarity_idx]  # 这里得到的是典故名称
        # 将典故名称转换为类型ID
        if max_similarity >= similarity_threshold:
            position_predictions[i] = 'B'
            type_predictions[i] = allusion_id.item()
            
            j = i + 1
            while j < seq_len:
                next_active_count = features['active_counts'][j].item()
                if next_active_count > 0:
                    next_values = features['values'][j][:next_active_count].numpy()
                    next_indices = features['indices'][j][:next_active_count].numpy()
                    next_max_similarity_idx = np.argmax(next_values)
                    next_max_similarity = next_values[next_max_similarity_idx]
                    next_allusion_id = next_indices[next_max_similarity_idx]
                    
                    if next_max_similarity >= similarity_threshold:
                        if next_allusion_id == allusion_id:
                            # 如果是相同的典故，标记为I
                            position_predictions[j] = 'I'
                            type_predictions[j] = allusion_id.item()
                            j += 1
                        else:
                            # 如果是新的典故，标记为B并更新当前典故ID
                            position_predictions[j] = 'B'
                            type_predictions[j] = next_allusion_id.item()
                            allusion_id = next_allusion_id
                            j += 1
                        continue
                break
            i = j
        else:
            i += 1
    
    return position_predictions, type_predictions

def test_position_and_type_prediction():
    """测试基于相似度的位置和类型预测"""
    # 加载典故词典和类型映射
    allusion_dict, _, _, num_types = load_allusion_dict()
    type2id={name: idx for idx, name in enumerate(allusion_dict.keys())}
    id2type={idx: name for name, idx in type2id.items()}
    
    # 加载特征和映射
    mapping_path =  'tests/allusion_mapping_MSM.json'
    features_path = 'tests/allusion_features_MSM.pt'
    
    sentence_to_id, _ = load_sentence_mappings(mapping_path)
    all_features = load_features(features_path)
    
    # 加载验证数据
    val_file = os.path.join(DATA_DIR, '4_test_position_no_bug_less_negatives.csv')
    val_data = pd.read_csv(val_file, sep='\t')
    
    # 位置标签统计
    position_true = []
    position_pred = []
    
    # 类别统计
    total_allusion_positions = 0  # 实际典故位置总数
    total_predicted_positions = 0  # 预测为典故的位置总数
    total_correct_predictions = 0  # 正确预测的位置总数
    
    with open('prediction_results.txt', 'w', encoding='utf-8') as f:
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
            true_positions = ['O'] * seq_len
            true_types = [0] * seq_len
            
            # 解析 transformed_allusion 列
            if pd.notna(row['transformed_allusion']):
                allusions = row['transformed_allusion'].split('];')
                for allusion in allusions:
                    if not allusion:
                        continue
                    # 移除开头的 [ 和结尾的 ]
                    allusion = allusion.strip('[]')
                    try:
                        # 找到最后一个数字后的逗号位置
                        parts = allusion.split(',')
                        positions = []
                        allusion_name = ''
                        
                        # 收集所有数字作为位置
                        for part in parts:
                            part = part.strip()
                            try:
                                pos = int(part)
                                positions.append(pos)
                            except ValueError:
                                # 不是数字的部分就是典故名称
                                allusion_name = part
                                break
                        
                        if positions and allusion_name:
                            # 使用type2id获取典故类型ID
                            allusion_type = type2id.get(allusion_name, 0)
                            
                            # 标记第一个位置为B，后续为I
                            true_positions[positions[0]] = 'B'
                            true_types[positions[0]] = allusion_type
                            for pos in positions[1:]:
                                true_positions[pos] = 'I'
                                true_types[pos] = allusion_type
                    except Exception as e:
                        print(f"Warning: 解析位置出错: {allusion}")
                        print(f"Error details: {str(e)}")
                        continue
            

            pred_positions, pred_types = predict_positions_and_types_by_similarity(
                features, type2id,id2type, similarity_threshold=0.76)
            
            # 收集位置标签
            position_true.extend(true_positions)
            position_pred.extend(pred_positions)
            
            # 统计类别结果
            for true_type, pred_type in zip(true_types, pred_types):
                if true_type != 0:
                    total_allusion_positions += 1
                if pred_type != 0:
                    total_predicted_positions += 1
                if true_type != 0 and true_type == pred_type:
                    total_correct_predictions += 1
            
            # 将结果写入文件
            f.write(f"sentence: {sentence}\n")
            f.write(f"true_positions: {true_positions}\n")
            f.write(f"true_types: {true_types}\n")
            f.write(f"pred_positions: {pred_positions}\n")
            f.write(f"pred_types: {pred_types}\n")
            f.write("\n")  # 添加空行分隔不同样本
            
            # 同时保持控制台输出
            print('sentence', sentence)
            print('true_positions', true_positions)
            print('true_types', true_types)
            print('pred_positions', pred_positions)
            print('pred_types', pred_types)
    
    # 计算位置标签的性能指标
    print("\n=== 位置标签识别统计（B-I-O）===")
    print(classification_report(position_true, position_pred, digits=4))
    
    # 计算类别识别的整体指标
    type_precision = total_correct_predictions / total_predicted_positions if total_predicted_positions > 0 else 0
    type_recall = total_correct_predictions / total_allusion_positions if total_allusion_positions > 0 else 0
    type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0
    
    print("\n=== 典故类别识别统计 ===")
    print(f"准确率: {type_precision:.4f}")
    print(f"召回率: {type_recall:.4f}")
    print(f"F1分数: {type_f1:.4f}")
    print(f"实际典故位置数: {total_allusion_positions}")
    print(f"预测典故位置数: {total_predicted_positions}")
    print(f"正确预测数: {total_correct_predictions}")

if __name__ == "__main__":
    test_position_and_type_prediction() 