import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
import difflib
import csv
from tqdm import tqdm
from collections import Counter

def merge_connected_spans(spans_str):
    try:
        spans = ast.literal_eval(spans_str)
    except:
        return spans_str
    
    if not spans:
        return spans
    
    # 将spans按起始位置排序
    spans.sort(key=lambda x: x[0])
    
    merged_spans = []
    current_positions = None
    
    for span in spans:
        if not isinstance(span, list) :
            continue
            
        # 获取当前span的所有位置
        current_span = list(range(span[0], span[-1] + 1))
        
        if current_positions is None:
            # 第一个span
            current_positions = current_span
            continue
            
        # 检查是否与当前positions相连
        if span[0] <= current_positions[-1] + 1:
            # 相连，合并
            # 使用集合去重，然后转回有序列表
            current_positions = sorted(list(set(current_positions + current_span)))
        else:
            # 不相连，保存当前positions并开始新的span
            merged_spans.append(current_positions)
            current_positions = current_span
    
    # 添加最后一组positions
    if current_positions:
        merged_spans.append(current_positions)
    
    return merged_spans

def balance_dataset(df, min_threshold=15, max_ratio=0.35):
    """
    平衡数据集，只对样本数特别少的典故进行补充
    Args:
        df: 输入的DataFrame
        min_threshold: 最小样本数阈值，低于此值的典故将被平衡
        max_ratio: 最大样本数与平均样本数的比率阈值
    """
    print("开始平衡数据集...")
    
    # 统计原始典故分布
    allusion_counts = Counter(df['allusion'].dropna())
    
    # 计算平均样本数和标准差
    mean_count = np.mean(list(allusion_counts.values()))
    target_count = int(mean_count * (1 + max_ratio))
    
    print("\n数据统计:")
    print(f"典故总数: {len(allusion_counts)}")
    print(f"平均样本数: {mean_count:.2f}")
    print(f"目标样本数: {target_count}")
    print(f"最小样本数阈值: {min_threshold}")
    
    # 准备平衡后的数据
    balanced_data = []
    
    # 对每个典故进行平衡
    for allusion, count in tqdm(allusion_counts.items(), desc="平衡数据"):
        # 获取当前典故的所有样本
        allusion_samples = df[df['allusion'] == allusion]
        balanced_data.append(allusion_samples)
        
        # 只对样本数过少的典故进行平衡
        if count < min_threshold:
            # 计算需要补充的样本数
            target = min(target_count, max(min_threshold, int(mean_count * 0.7)))
            samples_needed = target - count
            
            if samples_needed > 0:
                # 采样补充
                sampled = allusion_samples.sample(n=samples_needed, replace=True)
                balanced_data.append(sampled)
    
    # 合并所有采样数据
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    
    # 打印平衡后的统计信息
    balanced_counts = Counter(balanced_df['allusion'].dropna())
    print("\n平衡后统计:")
    print(f"平均样本数: {np.mean(list(balanced_counts.values())):.2f}")
    print(f"标准差: {np.std(list(balanced_counts.values())):.2f}")
    print(f"原始数据集大小: {len(df)}")
    print(f"平衡后数据集大小: {len(balanced_df)}")
    
    return balanced_df

# 使用difflib计算最长公共子串
def longest_common_substring(str1, str2):
    sequence_matcher = difflib.SequenceMatcher(None, str1, str2)
    match = sequence_matcher.find_longest_match(0, len(str1), 0, len(str2))
    return str1[match.a: match.a + match.size]

def merge_sentence_allusions(data):
    """合并同一诗句中的多个典故"""
    # 首先将allusion_index列中的字符串形式的列表转换为实际的列表
    data['allusion_index'] = data['allusion_index'].apply(lambda x: ast.literal_eval(str(x)) if x != '[]' else [])
    
    result = data.groupby('sentence').agg({
        'author': 'first',
        'title': 'first',
        'allusion': lambda x: ';'.join(x),
        'variation_number': 'sum',
        'allusion_index': lambda x: [item for sublist in x if sublist for item in (sublist if isinstance(sublist, list) else [sublist])]
    }).reset_index()
    
    def transform_allusion_index(row):
        allusions = row['allusion'].split(';')
        indices = row['allusion_index']
        
        # 如果indices是空列表，为每个典故添加空列表
        if not indices:
            # 修改这里的格式，确保是 [,典故] 而不是 [],典故
            return ';'.join([f"[,{allusion}]" for allusion in allusions])
        
        # 如果只有一个典故，确保indices是列表的列表
        if not isinstance(indices[0], list):
            indices = [indices]
            
        # 确保allusions和indices的长度匹配
        min_len = min(len(allusions), len(indices))
        combined = []
        
        for i in range(min_len):
            index_str = ','.join(map(str, indices[i])) if indices[i] else ''
            combined.append(f"[{index_str},{allusions[i]}]")
            
        # 处理剩余的典故（如果有）
        for i in range(min_len, len(allusions)):
            # 这里也修改格式
            combined.append(f"[,{allusions[i]}]")
            
        return ';'.join(combined)
    
    result['transformed_allusion'] = result.apply(transform_allusion_index, axis=1)
    return result

def process_variations(train_data, variation_data, prob_threshold=0.7):
    """处理训练集中未在异形词范围内的情况
    Args:
        train_data: 训练数据
        variation_data: 典故异形词数据
        prob_threshold: 将新异形词加入词表的概率阈值，默认0.7
    """
    for idx, row in tqdm(train_data.iterrows(), desc="处理异形词"):
        sentence = row['sentence']
        transformed_allusions = row['transformed_allusion'].split(';')
        new_transformed_allusions = []
        
        for allusion_info in transformed_allusions:
            inner_content = allusion_info.strip('[]')
            parts = inner_content.split(',')
            allusion = parts[-1]
            
            if len(parts) == 1 or (len(parts) == 2 and parts[0] == ''):
                variation_list = variation_data[variation_data['allusion'] == allusion]['variation_list'].values
                if len(variation_list) > 0:
                    variation_list = eval(variation_list[0])
                    merged_variations = ''.join(variation_list)
                    longest_match = longest_common_substring(merged_variations, sentence)
                    
                    if longest_match:
                        # 按概率决定是否将新异形词加入词表
                        if longest_match not in variation_list and np.random.random() < prob_threshold:
                            variation_list.append(longest_match)
                            variation_data.loc[
                                variation_data['allusion'] == allusion, 
                                'variation_list'
                            ] = str(variation_list)
                        
                        match_start = sentence.find(longest_match)
                        match_end = match_start + len(longest_match) - 1
                        positions = list(range(match_start, match_end + 1))
                        position_str = ','.join(map(str, positions))
                        new_transformed_allusions.append(f"[{position_str},{allusion}]")
                    else:
                        new_transformed_allusions.append(allusion_info)
                else:
                    new_transformed_allusions.append(allusion_info)
            else:
                new_transformed_allusions.append(allusion_info)
        
        train_data.at[idx, 'transformed_allusion'] = ';'.join(new_transformed_allusions)
    
    return variation_data, train_data

def process_data_pipeline():
    # 1. 读取和处理原始数据
    print("1. 读取原始数据...")
    df = pd.read_csv('data_strictly/爬取的典故数据_部分手工补全后.csv', sep='\t')
    df['allusion_index'] = df['allusion_index'].apply(merge_connected_spans)
    df.to_csv('data_strictly/1_spans_data.csv', sep='\t', index=False)

    # 2. 合并同一诗句的多个典故
    print("2. 合并同一诗句的多个典故...")
    merged_data = merge_sentence_allusions(df)
    merged_data.to_csv('data_strictly/2_merged_data.csv', sep='\t', index=False)
    merged_data = balance_dataset(merged_data)
    merged_data.to_csv('data_strictly/2_balanced_merged_data.csv', sep='\t', index=False)

    # 3. 划分训练/验证/测试集
    print("3. 划分数据集...")
    train_data, temp_data = train_test_split(merged_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    train_data.to_csv('data_strictly/3_train_data.csv', sep='\t', index=False)
    val_data.to_csv('data_strictly/3_val_data.csv', sep='\t', index=False)
    test_data.to_csv('data_strictly/3_test_data.csv', sep='\t', index=False)
    
    # 4. 处理训练集中未在异形词范围内的情况
    print("4. 处理训练集中的异形词...")
    variation_data = pd.read_csv('data_strictly/典故的异形数据.csv', sep='\t')
    updated_variation_data, updated_train_data = process_variations(merged_data, variation_data)
    
    # 保存更新后的数据
    updated_variation_data.to_csv('data_strictly/updated_典故的异形数据.csv', sep='\t', index=False)
    updated_train_data.to_csv('data_strictly/updated_train_data.csv', sep='\t', index=False)
    
    # 5. 生成类别识别任务的数据集
    print("5. 生成类别识别任务的数据集...")
    def split_to_single_allusion(data):
        """将包含多个典故的记录拆分为单个典故的记录"""
        records = []
        for _, row in data.iterrows():
            transformed_allusions = row['transformed_allusion'].split(';')
            for allusion_info in transformed_allusions:
                # 去掉方括号并分割
                inner_content = allusion_info.strip('[]')
                parts = inner_content.split(',')
                
                # 获取典故名称（最后一个元素）
                allusion = parts[-1]
                
                # 获取位置信息（如果有的话）
                if len(parts) > 1:
                    # 将位置信息转换为列表
                    indices = [int(x) for x in parts[:-1] if x]  # 排除空字符串
                    indices = str([indices]) if indices else '[[]]'
                else:
                    indices = '[[]]'
                
                records.append({
                    'sentence': row['sentence'],
                    'author': row['author'],
                    'title': row['title'],
                    'allusion': allusion,
                    'allusion_index': indices
                })
        return pd.DataFrame(records)
    
    train_type = split_to_single_allusion(train_data)
    val_type = split_to_single_allusion(val_data)
    test_type = split_to_single_allusion(test_data)
    
    # 6. 保存所有数据集
    print("6. 保存数据集...")
    # 位置识别任务的数据集
    train_data.to_csv('data_strictly/train_position.csv', sep='\t', index=False)
    val_data.to_csv('data_strictly/val_position.csv', sep='\t', index=False)
    test_data.to_csv('data_strictly/test_position.csv', sep='\t', index=False)
    
    # 类别识别任务的数据集
    train_type.to_csv('data_strictly/train_type.csv', sep='\t', index=False)
    val_type.to_csv('data_strictly/val_type.csv', sep='\t', index=False)
    test_type.to_csv('data_strictly/test_type.csv', sep='\t', index=False)
    
    print("数据处理完成！")

if __name__ == "__main__":
    process_data_pipeline()