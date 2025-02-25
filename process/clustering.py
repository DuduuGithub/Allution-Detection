import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from difflib import SequenceMatcher
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from process.process_config import OPTIMAL_EPS, min_samples_size
from tqdm import tqdm


def load_allusion_data(file_path='data/updated_典故的异性数据.csv'):
    """加载原始典故数据"""
    print(f"加载原始典故数据: {file_path}")
    df = pd.read_csv(file_path, encoding='utf-8', sep='\t')
    df.columns = df.columns.str.strip('"')
    return df


def merge_overlapping_clusters(clusters, potential_reps_dict):
    """合并重叠的聚类，但保留以潜在代表词为中心的重要聚类
    Args:
        clusters: 聚类结果字典
        potential_reps_dict: 每个初始聚类的潜在代表词字典
    """
    merged_clusters = {}
    used_words = set()
    cluster_id = 0
    
    # 获取所有潜在代表词
    all_potential_reps = set()
    for reps in potential_reps_dict.values():
        all_potential_reps.update(reps)
    
    # 首先处理包含潜在代表词的聚类
    rep_centered_clusters = []
    other_clusters = []
    
    for cluster_id, cluster in clusters.items():
        # 检查该聚类是否以潜在代表词为中心
        cluster_reps = cluster & all_potential_reps
        if cluster_reps:
            rep_centered_clusters.append((cluster_id, cluster, len(cluster_reps)))
        else:
            other_clusters.append((cluster_id, cluster))
    
    # 按照包含的代表词数量和聚类大小排序
    rep_centered_clusters.sort(key=lambda x: (x[2], len(x[1])), reverse=True)
    
    # 处理以代表词为中心的聚类
    new_cluster_id = 0
    for _, cluster, _ in rep_centered_clusters:
        valid_group = set()
        for word in cluster:
            if word not in used_words:
                valid_group.add(word)
        
        if len(valid_group) >= min_samples_size:
            merged_clusters[new_cluster_id] = valid_group
            used_words.update(valid_group)
            new_cluster_id += 1
    
    # 处理其他聚类
    for _, cluster in other_clusters:
        valid_group = set()
        for word in cluster:
            if word not in used_words:
                valid_group.add(word)
                
        if len(valid_group) >= min_samples_size:
            merged_clusters[new_cluster_id] = valid_group
            used_words.update(valid_group)
            new_cluster_id += 1
    
    return merged_clusters

def cluster_with_dbscan(variants, eps):
    """使用改进的聚类算法"""
    if not variants:
        return {}
    
    # 如果变体词数量小于等于3，每个词都作为独立的代表词
    if len(variants) <= 3:
        return {i: {word} for i, word in enumerate(variants)}
    
    # 计算距离矩阵
    n = len(variants)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            similarity = SequenceMatcher(None, variants[i], variants[j]).ratio()
            distance = 1 - similarity
            distances[i][j] = distances[j][i] = distance
    
    # 第一步：DBSCAN初始聚类
    min_samples = max(2, min(3, len(variants) // 3))
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distances)
    
    # 第一次聚类和细分
    initial_clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label == -1:
            continue
        if label not in initial_clusters:
            initial_clusters[label] = set()
        initial_clusters[label].add(variants[idx])
    
    def find_potential_representatives(cluster):
        """找出聚类中的潜在代表词"""
        # 方案1：基于中心度的代表词选择
        centrality_scores = {}
        for word1 in cluster:
            similar_words = 0
            total_similarity = 0
            for word2 in cluster:
                if word1 != word2:
                    similarity = SequenceMatcher(None, word1, word2).ratio()
                    if similarity >= (1 - eps):
                        similar_words += 1
                        total_similarity += similarity
            
            # 计算中心度分数：考虑相似词的数量和平均相似度
            if similar_words > 0:
                centrality_scores[word1] = (similar_words, total_similarity / similar_words)
        
        # 按照相似词数量和平均相似度排序
        sorted_words = sorted(
            centrality_scores.items(),
            key=lambda x: (x[1][0], x[1][1]),  # 先按相似词数量，再按平均相似度
            reverse=True
        )
        
        return [word for word, _ in sorted_words[:3]]  # 返回top3潜在代表词
    
    # 记录每个初始聚类的潜在代表词
    potential_reps_dict = {}
    
    # 对每个初始聚类进行细分
    intermediate_clusters = {}
    cluster_id = 0
    
    for initial_cluster_id, cluster in initial_clusters.items():
        # 构建词之间的相似关系图
        word_connections = {word: set() for word in cluster}
        for word1 in cluster:
            for word2 in cluster:
                if word1 != word2:
                    similarity = SequenceMatcher(None, word1, word2).ratio()
                    if similarity >= (1 - eps):
                        word_connections[word1].add(word2)
        
        # 找出潜在的代表词并记录
        potential_reps = find_potential_representatives(cluster)
        potential_reps_dict[initial_cluster_id] = set(potential_reps)
        
        # 从潜在代表词开始进行分割
        processed = set()
        for rep in potential_reps:
            # 不管是否processed都尝试形成新组
            current_group = {rep}
            to_check = word_connections[rep].copy()
            
            while to_check:
                next_word = to_check.pop()
                if next_word not in current_group:
                    all_similar = all(
                        SequenceMatcher(None, next_word, w).ratio() >= (1 - eps)
                        for w in current_group
                    )
                    if all_similar:
                        current_group.add(next_word)
                        to_check.update(
                            w for w in word_connections[next_word] 
                            if w not in current_group and w not in to_check
                        )
            if len(current_group) >= min_samples_size:
                intermediate_clusters[cluster_id] = current_group
                cluster_id += 1
                processed.update(current_group)
        
        # 处理剩余未分配的词
        remaining_words = cluster - processed
        if remaining_words:
            for word in remaining_words:
                if word in processed:
                    continue
                
                current_group = {word}
                to_check = word_connections[word].copy()
                
                while to_check:
                    next_word = to_check.pop()
                    if next_word not in current_group:
                        all_similar = all(
                            SequenceMatcher(None, next_word, w).ratio() >= (1 - eps)
                            for w in current_group
                        )
                        if all_similar:
                            current_group.add(next_word)
                            to_check.update(
                                w for w in word_connections[next_word] 
                                if w not in current_group and w not in to_check
                            )
                
                if len(current_group) >= min_samples_size:
                    intermediate_clusters[cluster_id] = current_group
                    cluster_id += 1
                    processed.update(current_group)
    
    # 合并重叠的聚类，传入潜在代表词信息
    final_clusters = merge_overlapping_clusters(intermediate_clusters, potential_reps_dict)
    
    return final_clusters

def select_representative(cluster):
    """为一个聚类选择代表词"""
    if not cluster:
        return None
    
    # 如果只有一个元素，直接返回
    if len(cluster) == 1:
        return list(cluster)[0]
    
    # 计算每个变体与其他变体的平均相似度
    similarity_scores = {}
    for variant1 in cluster:
        total_similarity = 0
        for variant2 in cluster:
            if variant1 != variant2:
                # 计算相似度
                similarity = SequenceMatcher(None, variant1, variant2).ratio()
                total_similarity += similarity
        avg_similarity = total_similarity / (len(cluster) - 1)
        similarity_scores[variant1] = avg_similarity
    
    # 选择平均相似度最高的变体作为代表词
    best_representative = max(similarity_scores.items(), key=lambda x: x[1])[0]
    return best_representative


def process_all_allusions():
    """处理所有典故的聚类并保存结果"""
    # 加载原始数据
    print("加载原始数据...")
    df = pd.read_csv('data/updated_典故的异性数据.csv', encoding='utf-8', sep='\t')
    
    # 存储所有处理结果
    all_results = []
    
    # 使用tqdm包装迭代器来显示进度
    for _, row in tqdm(df.iterrows(), total=len(df), desc="处理典故"):
        allusion_name = row['allusion']
        variants = eval(row['variation_list'])
                        
        # 使用聚类方法
        clusters = cluster_with_dbscan(variants, OPTIMAL_EPS)
        
        # 收集该典故的所有代表词
        representatives = []
        variations_dict = {}  # 存储每个代表词对应的变体
        
        # 对每个聚类选择代表词
        for cluster_id, cluster_words in clusters.items():
            representative = select_representative(cluster_words)
            representatives.append(representative)
            variations_dict[representative] = list(cluster_words)
        
        # 保存结果
        all_results.append({
            'allusion': allusion_name,
            'representatives': representatives,  # 所有代表词列表
            'variations_dict': variations_dict  # 每个代表词对应的变体字典
        })
    
    # 将结果转换为DataFrame并保存
    print("\n保存结果...")
    result_df = pd.DataFrame(all_results)
    
    # 将字典和列表转换为字符串形式保存
    result_df['representatives'] = result_df['representatives'].apply(str)
    result_df['variations_dict'] = result_df['variations_dict'].apply(str)
    
    result_df.to_csv('data/cleared_allusion_type.csv', 
                     index=False, 
                     encoding='utf-8', 
                     sep='\t')
    
    print("\n处理完成!")
    print(f"总共处理了 {len(df)} 个典故")
    print(f"结果已保存到: data/cleared_allusion_type.csv")

if __name__ == "__main__":
    process_all_allusions() 