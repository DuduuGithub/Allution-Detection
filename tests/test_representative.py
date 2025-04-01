'''
    统计生成的代表词识别出的异形词的准确率
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from difflib import SequenceMatcher
# from process.process_config import OPTIMAL_EPS
import random
from tqdm import tqdm

from process.process_config import OPTIMAL_EPS
from process.clustering import load_allusion_data

def load_representative_dict(file_path='data/allusion_type.csv'):
    """加载代表词字典"""
    df = pd.read_csv(file_path, encoding='utf-8', sep='\t')
    rep_dict = {}
    for _, row in df.iterrows():
        allusion = row['allusion']
        # 将字符串形式的列表转换为实际的列表
        representatives = eval(row['representatives'])
        rep_dict[allusion] = representatives
    return rep_dict

def load_representative_dict_large(file_path='data/updated_典故的异性数据.csv'):
    """加载代表词字典"""
    df = pd.read_csv(file_path, encoding='utf-8', sep='\t')
    rep_dict = {}
    for _, row in df.iterrows():
        allusion = row['allusion']
        # 将字符串形式的列表转换为实际的列表
        representatives = eval(row['variation_list'])
        rep_dict[allusion] = representatives
    return rep_dict

def test_variant_recognition(sample_size=None):
    """测试变体识别效果"""
    # 加载数据
    print("加载数据...")
    original_df = load_allusion_data()
    rep_dict = load_representative_dict_large()
    
    print(f"\n使用的eps阈值: {OPTIMAL_EPS}")
    
    # 选择要测试的典故
    allusions = list(rep_dict.keys()) if sample_size is None else \
                random.sample(list(rep_dict.keys()), min(sample_size, len(rep_dict)))
    
    print(f"\n=== 测试 {len(allusions)} 个典故 ===")
    
    total_variants = 0
    total_correct = 0
    error_cases = []  # 存储错误匹配的案例
    allusion_errors = {}  # 存储每个典故的错误次数
    
    for allusion_name in tqdm(allusions, desc="处理典故"):
        # 获取该典故的原始变体
        allusion_data = original_df[original_df['allusion'] == allusion_name]
        if len(allusion_data) == 0:
            continue
            
        try:
            variants = eval(allusion_data.iloc[0]['variation_list'])
        except:
            continue
            
        if not variants:
            continue
        
        # 测试每个变体
        allusion_error_count = 0  # 记录当前典故的错误次数
        
        for variant in variants:
            total_variants += 1
            matched_allusions = {}  # 使用字典存储匹配到的典故及其最高相似度
            
            # 检查所有典故的代表词
            for test_allusion, representatives in rep_dict.items():
                # 计算与所有代表词的相似度
                max_similarity = 0
                best_rep = None
                for rep in representatives:
                    similarity = SequenceMatcher(None, variant, rep).ratio()
                    if similarity > max_similarity and (1 - similarity) < OPTIMAL_EPS:
                        max_similarity = similarity
                        best_rep = rep
                
                if best_rep:
                    matched_allusions[test_allusion] = (best_rep, max_similarity)
            
            # 如果原典故在匹配到的典故中，则判定为正确
            if allusion_name in matched_allusions:
                total_correct += 1
            else:
                # 记录错误案例
                error_case = {
                    'variant': variant,
                    'original_allusion': allusion_name,
                    'matched_allusions': matched_allusions
                }
                error_cases.append(error_case)
                allusion_error_count += 1
        
        if allusion_error_count > 0:
            allusion_errors[allusion_name] = allusion_error_count
    
    # 打印总体统计
    print("\n=== 总体统计 ===")
    print(f"测试变体总数: {total_variants}")
    print(f"正确匹配数: {total_correct}")
    print(f"准确率: {total_correct/total_variants*100:.2f}%")
    
    # 打印错误率最高的典故
    print("\n=== 错误率最高的典故 ===")
    sorted_errors = sorted(allusion_errors.items(), key=lambda x: x[1], reverse=True)
    for allusion, error_count in sorted_errors[:10]:
        total = len(eval(original_df[original_df['allusion'] == allusion].iloc[0]['variation_list']))
        error_rate = error_count / total * 100
        print(f"\n典故: {allusion}")
        print(f"错误数: {error_count}/{total} ({error_rate:.1f}%)")
    
    # 打印错误案例
    print("\n=== 错误匹配案例 ===")
    print(f"错误案例数量: {len(error_cases)}")
    
    # 随机选择10个错误案例进行展示
    sample_size = min(10, len(error_cases))
    if sample_size > 0:
        print(f"\n随机展示 {sample_size} 个错误案例:")
        for case in random.sample(error_cases, sample_size):
            print("\n---")
            print(f"变体词: {case['variant']}")
            print(f"原典故: {case['original_allusion']}")
            print("匹配到的典故:")
            # 按相似度排序显示所有匹配到的典故
            sorted_matches = sorted(case['matched_allusions'].items(), 
                                 key=lambda x: x[1][1], 
                                 reverse=True)
            for matched_allusion, (rep, similarity) in sorted_matches:
                print(f"  - {matched_allusion}")
                print(f"    代表词: {rep}")
                print(f"    相似度: {similarity:.3f}")

if __name__ == "__main__":
    # 测试所有典故的变体识别效果
    test_variant_recognition(100)
