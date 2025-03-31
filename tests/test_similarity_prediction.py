'''
    测试给出target_position,仅仅进行相似度判别的准确性与误报率。仅看top1
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from model.config import DATA_DIR
from process.process_dict_features import load_sentence_mappings, load_features
from model.train import load_allusion_dict
from collections import defaultdict
from sklearn.metrics import classification_report
import random

def vote_for_allusion(features, start, end, id_to_allusion, similarity_threshold=0.5, vote_ratio_threshold=0.3):
    """
    对一个片段进行典故投票
    返回: (预测的典故, 最大相似度, 是否有效预测, 投票统计信息)
    """
    # 收集所有位置的典故投票
    allusion_votes = defaultdict(lambda: {'count': 0, 'total_similarity': 0.0})
    total_features = 0
    
    # 遍历片段中的每个位置
    for pos in range(start, end + 1):
        active_count = features['active_counts'][pos].item()
        total_features += active_count
        
        for idx in range(active_count):
            allusion_id = features['indices'][pos][idx].item()
            similarity = features['values'][pos][idx].item()
            allusion_name = id_to_allusion[allusion_id]
            
            allusion_votes[allusion_name]['count'] += 1
            allusion_votes[allusion_name]['total_similarity'] += similarity

    if not allusion_votes:
        return None, 0.0, False, {}

    # 计算每个典故的平均相似度
    for allusion in allusion_votes.values():
        allusion['avg_similarity'] = allusion['total_similarity'] / allusion['count']

    # 选择得票最多且平均相似度最高的典故
    best_allusion = max(
        allusion_votes.items(),
        key=lambda x: (x[1]['count'], x[1]['avg_similarity'])
    )
    
    allusion_name = best_allusion[0]
    avg_similarity = best_allusion[1]['avg_similarity']
    vote_ratio = best_allusion[1]['count'] / total_features if total_features > 0 else 0
    
    # 判断是否是有效预测
    is_valid = avg_similarity >= similarity_threshold and vote_ratio >= vote_ratio_threshold
    
    # 准备投票统计信息
    vote_stats = {
        'total_features': total_features,
        'vote_ratio': vote_ratio,
        'votes': {name: info for name, info in allusion_votes.items()}
    }
    
    return allusion_name, avg_similarity, is_valid, vote_stats

def generate_negative_positions(sentence_length, existing_positions, num_samples=3, window_size=5):
    """
    为无典故句子生成随机的连续位置序列作为反例
    Args:
        sentence_length: 句子长度
        existing_positions: 已存在的典故位置列表
        num_samples: 要生成的样本数量
        window_size: 窗口大小
    """
    negative_positions = []
    existing_indices = set()
    
    # 将已存在的位置标记为占用
    for pos_list in existing_positions:
        for pos in pos_list:
            existing_indices.add(pos)
    
    # 生成随机位置
    attempts = 0
    while len(negative_positions) < num_samples and attempts < 100:
        # 随机选择起始位置
        start = random.randint(0, max(0, sentence_length - window_size))
        end = min(start + window_size - 1, sentence_length - 1)
        
        # 检查是否与已有位置重叠
        valid = True
        for i in range(start, end + 1):
            if i in existing_indices:
                valid = False
                break
        
        if valid:
            negative_positions.append([start, end])
        
        attempts += 1
    
    return negative_positions

def test_similarity_based_prediction():
    """测试基于相似度的典故预测"""
    # 加载映射和特征文件
    mapping_path = os.path.join(DATA_DIR, 'allusion_mapping_strictly_dict.json')
    features_path = os.path.join(DATA_DIR, 'allusion_features_strictly_dict.pt')
    
    sentence_to_id, id_to_sentence = load_sentence_mappings(mapping_path)
    all_features = load_features(features_path)
    
    # 加载典故词典用于反查
    allusion_dict, type_label2id, id2type_label, _ = load_allusion_dict()
    allusion_to_id = {name: idx for idx, name in enumerate(allusion_dict.keys())}
    id_to_allusion = {idx: name for name, idx in allusion_to_id.items()}
    
    # 加载验证集数据和全量数据
    val_file = os.path.join(DATA_DIR, '4_val_type_no_bug.csv')
    full_file = os.path.join(DATA_DIR, '3_1_2_final_position_dataset.csv')
    
    val_data = pd.read_csv(val_file, sep='\t')
    full_data = pd.read_csv(full_file, sep='\t')
    
    # 从全量数据中筛选出无典故的样本
    non_allusion_data = full_data[full_data['variation_number'] == 0].sample(n=min(len(val_data), 100), random_state=42)
    
    # 合并数据集
    test_data = pd.concat([val_data, non_allusion_data], ignore_index=True)
    
    # 统计变量
    overall_stats = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    
    # 处理每个样本
    for _, row in test_data.iterrows():
        sentence = row['sentence']
        variation_number = row['variation_number']
        
        if sentence not in sentence_to_id:
            continue
            
        sent_id = sentence_to_id[sentence]
        features = all_features[sent_id]
        
        # 根据 variation_number 判断是否包含典故
        if variation_number == 0:
            # 无典故样本，生成随机位置作为测试位置
            sentence_length = len(sentence)
            test_positions = generate_negative_positions(sentence_length, [], num_samples=3)
            true_allusion = "O"
        else:
            # 有典故样本，使用标注的位置
            try:
                allusion_positions = eval(str(row['allusion_index'])) if pd.notna(row['allusion_index']) else []
                test_positions = [[min(positions), max(positions)] for positions in allusion_positions]
                true_allusion = row['allusion'] if pd.notna(row['allusion']) else "O"
            except:
                continue
        
        # 确保至少有一个测试位置
        if not test_positions:
            continue
        
        # 对每个位置进行预测
        for start, end in test_positions:
            predicted_allusion, similarity, is_valid, vote_stats = vote_for_allusion(
                features, start, end, id_to_allusion, 
                similarity_threshold=0.5,
                vote_ratio_threshold=0.3
            )
            
            # 确定预测类型
            pred_type = ""
            if true_allusion == "O":
                if not is_valid:
                    pred_type = "TN"
                    overall_stats["TN"] += 1
                else:
                    pred_type = "FP"
                    overall_stats["FP"] += 1
            else:
                if is_valid and predicted_allusion == true_allusion:
                    pred_type = "TP"
                    overall_stats["TP"] += 1
                elif not is_valid:
                    pred_type = "FN"
                    overall_stats["FN"] += 1
                else:
                    pred_type = "FP+FN"
                    overall_stats["FP"] += 1
                    overall_stats["FN"] += 1
            
            # 打印样本信息
            print("\n=== 样本详情 ===")
            print(f"诗句: {sentence}")
            print(f"位置: [{start}, {end}]")
            print(f"真实典故: {true_allusion}")
            print(f"预测典故: {predicted_allusion if is_valid else 'O'}")
            print(f"预测类型: {pred_type}")
            print(f"相似度: {similarity:.4f}")
            if vote_stats:  # 添加检查以确保 vote_stats 不为空
                print(f"总特征数: {vote_stats['total_features']}")
                print(f"投票比例: {vote_stats['votes'][predicted_allusion]['count'] / vote_stats['total_features']:.4f}" if predicted_allusion else "投票比例: 0.0000")
    
    # 计算整体指标
    tp = overall_stats['TP']
    fp = overall_stats['FP']
    fn = overall_stats['FN']
    tn = overall_stats['TN']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print("\n=== 整体评估指标 ===")
    print(f"准确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    test_similarity_based_prediction()