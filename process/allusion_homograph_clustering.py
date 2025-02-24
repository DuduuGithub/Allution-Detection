from difflib import SequenceMatcher
import numpy as np
from sklearn.cluster import DBSCAN
import json
import pandas as pd

# 计算两个字符串之间的相似度
def string_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

# 选择每个聚类的代表词
def select_representative(cluster):
    # 计算每个词与其他词的平均相似度
    avg_similarities = {}
    for word1 in cluster:
        total_sim = 0
        for word2 in cluster:
            if word1 != word2:
                total_sim += string_similarity(word1, word2)
        avg_similarities[word1] = total_sim / (len(cluster) - 1) if len(cluster) > 1 else 0
    
    # 选择与其他词平均相似度最高的词作为代表词
    return max(avg_similarities.items(), key=lambda x: x[1])[0]

def calculate_optimal_eps_for_group(grouped_words):
    """为单个大组计算最优eps值"""
    # 存储所有组内相似度和组间相似度
    intra_similarities = []  # 组内相似度
    inter_similarities = []  # 组间相似度
    
    # 获取每个小组的代表词
    representatives = []
    for group in grouped_words:
        representatives.append(select_representative(group))
    
    # 计算组内相似度
    for i, group in enumerate(grouped_words):
        rep = representatives[i]
        for word in group:
            if word != rep:
                intra_similarities.append(string_similarity(rep, word))
    
    # 计算组间相似度（仅在同一大组内的小组之间）
    for i in range(len(grouped_words)):
        for j in range(i + 1, len(grouped_words)):
            rep_i = representatives[i]
            for word in grouped_words[j]:
                inter_similarities.append(string_similarity(rep_i, word))
    
    # 将相似度转换为numpy数组并排序
    intra_similarities = np.array(intra_similarities)
    inter_similarities = np.array(inter_similarities)
    
    # 合并所有相似度并排序，用作候选eps值
    all_similarities = np.unique(np.concatenate([intra_similarities, inter_similarities]))
    all_similarities.sort()
    
    # 寻找最优eps
    best_eps = None
    best_score = -1
    
    for eps in all_similarities:
        # 计算在当前eps下的指标
        intra_included = np.sum(intra_similarities >= eps)  # 包含的组内相似度数量
        inter_excluded = np.sum(inter_similarities < eps)   # 排除的组间相似度数量
        
        # 计算准确率和召回率
        intra_recall = intra_included / len(intra_similarities) if len(intra_similarities) > 0 else 0
        inter_precision = inter_excluded / len(inter_similarities) if len(inter_similarities) > 0 else 0
        
        # 计算F1分数
        if intra_recall + inter_precision > 0:
            f1_score = 2 * (intra_recall * inter_precision) / (intra_recall + inter_precision)
        else:
            f1_score = 0
            
        # 更新最优值
        if f1_score > best_score:
            best_score = f1_score
            best_eps = eps
    
    return best_eps if best_eps is not None else 0.5

def calculate_optimal_eps(major_groups):
    """计算所有大组的平均最优eps值
    
    Args:
        major_groups: 列表的列表的列表，结构为 [大组[小组[词]]]
    Returns:
        float: 平均最优eps值
    """
    eps_values = []
    
    # 为每个大组计算最优eps
    for major_group in major_groups:
        eps = calculate_optimal_eps_for_group(major_group)
        if eps is not None:
            eps_values.append(eps)
            print(f"Eps for major group: {eps:.3f}")
    
    return np.mean(eps_values)

def cluster_with_dbscan(words, eps):
    # 构建相似度矩阵
    n = len(words)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i][j] = string_similarity(words[i], words[j])
    
    # 使用DBSCAN进行聚类
    # 距离矩阵 = 1 - 相似度矩阵
    distance_matrix = 1 - similarity_matrix
    dbscan = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
    clusters = dbscan.fit_predict(distance_matrix)
    
    # 整理聚类结果
    cluster_dict = {}
    for i, label in enumerate(clusters):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(words[i])
    
    return cluster_dict

def save_optimal_eps(eps_value, output_path='data/optimal_eps.json'):
    """保存最优eps值"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'optimal_eps': eps_value}, f, ensure_ascii=False)

def process_allusion_variants(input_file, output_file, eps):
    """处理典故异形词数据，生成代表词组文件
    
    Args:
        input_file: 输入的异形词CSV文件路径
        output_file: 输出的代表词组CSV文件路径
        eps: 聚类阈值
    """
    # 读取CSV文件
    df = pd.read_csv(input_file, encoding='utf-8')
    
    # 按典故ID分组处理
    results = []
    for _, group in df.groupby('allusion_id'):
        variants = group['variant'].tolist()
        
        # 使用DBSCAN聚类
        clusters = cluster_with_dbscan(variants, eps)
        
        # 为每个聚类选择代表词
        for cluster in clusters.values():
            representative = select_representative(cluster)
            # 保存结果
            results.append({
                'representative': representative,
                'variants': ';'.join(cluster)
            })
    
    # 保存到CSV
    pd.DataFrame(results).to_csv(output_file, index=False, encoding='utf-8')

if __name__ == "__main__":
    # 示例分组（多个大组，每个大组包含多个小组）
    major_groups = [
        # 第一个大组
        [
            ['开三径', '三径', '开竹径'],
            ['羊求', '求仲', '羊仲', '求羊径'],
            ['蒋诩径', '蒋生径']
        ],
        # 第二个大组
        [
            ['其他小组1的词列表'],
            ['其他小组2的词列表']
        ]
    ]
    
    # 1. 计算最优eps
    optimal_eps = calculate_optimal_eps(major_groups)
    print(f"\n请将以下值添加到 model/config.py 中：")
    print(f"OPTIMAL_EPS = {optimal_eps:.3f}")
    
    # 2. 处理异形词数据
    process_allusion_variants(
        input_file='data/updated_典故的异性数据.csv',
        output_file='data/典故的代表词组.csv',
        eps=optimal_eps
    )
