import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch
from difflib import SequenceMatcher
from model.config import OPTIMAL_EPS

'''
    输入诗句，输出典故特征提取结果
'''


def prepare_sparse_features(batch_texts, allusion_dict, max_active=5):
    """将文本批量转换为稀疏特征格式
    
    对每个起始位置，尝试长度2~5的窗口，找出最佳匹配的典故
    """
    batch_size = len(batch_texts)
    seq_len = max(len(text) for text in batch_texts)
    
    # 初始化每个位置的典故匹配列表
    position_to_allusions = [[] for _ in range(seq_len)]  # 每个位置可能的典故列表
    
    # 为每个典故分配ID
    allusion_to_id = {name: idx for idx, name in enumerate(allusion_dict.keys())}
    
    def get_variants(variants):
        """获取变体列表"""
        if isinstance(variants, list):
            # 如果是列表但只有一个元素且是字符串，可能需要进一步解析
            if len(variants) == 1 and isinstance(variants[0], str):
                try:
                    # 尝试解析可能的嵌套列表
                    parsed = eval(variants[0])
                    if isinstance(parsed, list):
                        return parsed
                except:
                    pass
            return variants
            
        if isinstance(variants, str):
            try:
                # 尝试解析字符串形式的列表
                parsed = eval(variants)
                if isinstance(parsed, list):
                    # 如果解析出的是列表，递归处理可能的嵌套
                    return get_variants(parsed)
                return [variants]
            except:
                # 如果解析失败，按分号分割
                return variants.split(';')
                
        return []  # 兜底返回空列表
    
    # 处理每个样本
    for b, text in enumerate(batch_texts):
        # 对每个起始位置
        for start_pos in range(len(text)-1):
            max_len = min(5, len(text) - start_pos)
            
            # 尝试不同长度的窗口
            for window_size in range(2, max_len + 1):
                context = text[start_pos:start_pos + window_size]
                end_pos = start_pos + window_size - 1
                
                # 匹配典故
                for allusion_name, variants in allusion_dict.items():
                    variant_list = get_variants(variants)
                    max_similarity = 0
                    
                    # 找出该典故在当前位置的最高相似度
                    for variant in variant_list:
                        similarity = SequenceMatcher(None, context, variant).ratio()
                        if similarity > max_similarity:
                            max_similarity = similarity
                            # print(f"字段: {context}, 典故: {allusion_name}, 变体: {variant}, 相似度: {max_similarity:.3f}")
                    
                    
                    # 计算距离
                    distance = 1 - max_similarity
                    if distance < OPTIMAL_EPS:
                        # 将该典故添加到范围内的每个位置
                        for pos in range(start_pos, end_pos + 1):
                            position_to_allusions[pos].append((
                                allusion_to_id[allusion_name],
                                max_similarity
                            ))
        
        # 为每个位置选择最佳的max_active个典故
        indices = torch.zeros((batch_size, seq_len, max_active), dtype=torch.long)
        values = torch.zeros((batch_size, seq_len, max_active), dtype=torch.float)
        active_counts = torch.zeros((batch_size, seq_len), dtype=torch.long)
        
        for pos in range(seq_len):
            # 对该位置的所有典故按相似度排序
            matches = position_to_allusions[pos]
            matches.sort(key=lambda x: x[1], reverse=True)
            matches = matches[:max_active]  # 只保留最佳的max_active个
            
            # 记录匹配数量和结果
            active_counts[b, pos] = len(matches)
            for idx, (allusion_id, similarity) in enumerate(matches):
                indices[b, pos, idx] = allusion_id
                values[b, pos, idx] = similarity
    
    return {
        'indices': indices,
        'values': values,
        'active_counts': active_counts
    }

# 经测试，使用updated_典故的异性数据.csv 效果会更好，比如对桃源避秦人不见，武陵渔父独知处。 识别不出桃源 说明判断相关性的算法可能需要再优化。二者的时间差异还挺大的
def load_representative_dict(file_path='data/allusion_representative.csv'):
    """加载典故代表词数据"""
    print(f"加载典故代表词数据: {file_path}")
    df = pd.read_csv(file_path, encoding='utf-8',sep='\t')
    
    allusion_dict = {}
    for _, row in df.iterrows():
        allusion = row['allusion']
        representatives = row['variation_list'].split('\t')
        allusion_dict[allusion] = representatives
    
    return allusion_dict

def calculate_similarity(s1, s2):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, s1, s2).ratio()

def test_poem_recognition():
    """测试诗句中的典故识别效果"""
    test_cases = [
        ("一入石渠署，三闻宫树蝉。", [(2, 3,'石渠')]),
        ("桃源避秦人不见，武陵渔父独知处。", [(0, 2, "桃源")]),
    ]
    
    # 加载典故词典
    allusion_dict = load_representative_dict()
    # 创建ID到典故名的映射
    # 为每个典故分配ID
    allusion_to_id = {name: idx for idx, name in enumerate(allusion_dict.keys())}
    id_to_allusion = {idx: name for name, idx in allusion_to_id.items()}
    
    print(f"\n使用的eps阈值: {OPTIMAL_EPS}")
    
    for poem, expected in test_cases:
        print(f"\n诗句: {poem}")
        print(f"预期典故: {expected}")
        
        # 获取特征
        features = prepare_sparse_features([poem], allusion_dict)
        
        print("\n原始特征张量:")
        print("indices shape:", features['indices'].shape)
        print("indices:\n", features['indices'][0])  # [seq_len, max_active]
        print("values:\n", features['values'][0])    # [seq_len, max_active]
        print("active_counts:\n", features['active_counts'][0])  # [seq_len]
        
        print("\n每个位置检测到的典故：")
        for pos in range(len(poem)):
            active_count = features['active_counts'][0][pos].item()
            if active_count > 0:
                print(f"\n位置 {pos} ({poem[pos]}):")
                for idx in range(active_count):
                    allusion_id = features['indices'][0][pos][idx].item()
                    similarity = features['values'][0][pos][idx].item()
                    allusion_name = id_to_allusion[allusion_id]
                    print(f"  - 典故: {allusion_name}, 相似度: {similarity:.3f}")
        print("\n" + "="*50)

def main():
    try:
        test_poem_recognition()
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 