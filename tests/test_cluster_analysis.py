import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from difflib import SequenceMatcher
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from process.process_config import OPTIMAL_EPS,min_samples_size
'''
    输入典故名，输出该典故的聚类结果
'''

def load_allusion_data(file_path='data/updated_典故的异性数据.csv'):
    """加载原始典故数据"""
    print(f"加载原始典故数据: {file_path}")
    df = pd.read_csv(file_path, encoding='utf-8', sep='\t')
    df.columns = df.columns.str.strip('"')
    return df

def analyze_cluster_quality(cluster_words, representative):
    """分析聚类质量"""
    similarities = []
    for word in cluster_words:
        if word != representative:
            similarity = SequenceMatcher(None, word, representative).ratio()
            similarities.append((word, similarity))
    return sorted(similarities, key=lambda x: x[1], reverse=True)

def merge_overlapping_clusters(clusters, potential_reps_dict):
    """合并重叠的聚类，但保留以潜在代表词为中心的重要聚类"""
    merged_clusters = {}
    used_words = set()
    
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
        
        if valid_group: 
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
    
    # 处理剩余的未使用词（包括潜在代表词）
    remaining_words = set(word for cluster in clusters.values() for word in cluster) - used_words
    for word in remaining_words:
        merged_clusters[new_cluster_id] = {word}
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
        print(f"聚类{cluster}潜在代表词: {potential_reps}")
        
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
            print(f"潜在代表词 新组: {current_group}")
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
                
                if len(current_group) >= min_samples:
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

def test_specific_allusion_clustering(allusion_name="桃源（陶潜）"):
    """测试特定典故的聚类效果"""
    # 加载数据
    original_df = load_allusion_data()
    
    # 获取该典故的原始变体
    allusion_data = original_df[original_df['allusion'] == allusion_name]
    if len(allusion_data) == 0:
        print(f"未找到典故: {allusion_name}")
        return
        
    variants = eval(allusion_data.iloc[0]['variation_list'])
    if not variants:
        print("该典故没有变体数据")
        return
    
    print(f"\n=== 分析典故: {allusion_name} ===")
    print(f"原始变体数量: {len(variants)}")
    
    # 使用改进的聚类方法
    clusters = cluster_with_dbscan(variants, OPTIMAL_EPS)
    noise_points = [v for v in variants if not any(v in c for c in clusters.values())]
    
    # 分析并保存结果
    results = []
    
    # 分析每个聚类
    print("\n=== 聚类结果 ===")
    for label, cluster_words in clusters.items():
        print(f"\n聚类 {label + 1}:")
        print(f"成员数量: {len(cluster_words)}")
        
        # 使用改进的代表词选择方法
        best_rep = select_representative(cluster_words)
        
        # 分析该聚类的质量
        quality_analysis = analyze_cluster_quality(cluster_words, best_rep)
        
        # 保存结果
        results.append({
            'cluster_id': label + 1,
            'size': len(cluster_words),
            'representative': best_rep,
            'members': cluster_words,
            'similarities': quality_analysis
        })
        
        # 打印详细信息
        print(f"代表词: {best_rep}")
        print("成员相似度:")
        for word, sim in quality_analysis:
            print(f"  - {word}: {sim:.3f}")
    
    # 处理噪声点
    if noise_points:
        print("\n噪声点:")
        for point in noise_points:
            print(f"  - {point}")
        
        results.append({
            'cluster_id': 'noise',
            'size': len(noise_points),
            'representative': None,
            'members': noise_points,
            'similarities': []
        })
    
    # 保存分析结果
    output_file = f'analysis_{allusion_name}_clustering.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"典故聚类分析: {allusion_name}\n")
        f.write(f"原始变体数量: {len(variants)}\n")
        f.write(f"聚类数量: {len(clusters)}\n")
        f.write(f"噪声点数量: {len(noise_points)}\n\n")
        
        for result in results:
            if result['cluster_id'] == 'noise':
                f.write("\n=== 噪声点 ===\n")
                for point in result['members']:
                    f.write(f"- {point}\n")
            else:
                f.write(f"\n=== 聚类 {result['cluster_id']} ===\n")
                f.write(f"代表词: {result['representative']}\n")
                f.write(f"成员数量: {result['size']}\n")
                f.write("成员相似度:\n")
                for word, sim in result['similarities']:
                    f.write(f"- {word}: {sim:.3f}\n")
    
    print(f"\n分析结果已保存到: {output_file}")

def main():
    try:
        print("min_samples_size:",min_samples_size)
        test_specific_allusion_clustering("焦尾（蔡邕）")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 