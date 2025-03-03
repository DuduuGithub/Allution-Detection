import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
import difflib
import csv
from tqdm import tqdm

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

# 读取CSV文件
df = pd.read_csv('爬取的典故数据_部分手工补全后.csv', sep='\t')

# 处理allusion_index列
df['allusion_index'] = df['allusion_index'].apply(merge_connected_spans)

# 保存处理后的数据
df.to_csv('1_process_connected_spans_data.csv', sep='\t', index=False)

# 打印一些示例进行验证
print("处理后的一些示例：")
for i, row in df.iterrows():
    if isinstance(row['allusion_index'], list) and len(row['allusion_index']) > 1:
        print(f"原句：{row['sentence']}")
        print(f"典故：{row['allusion']}")
        print(f"处理后的spans：{row['allusion_index']}")
        print("-" * 50)

# 测试代码
test_cases = [
    "[[1, 2], [4, 5]]",  # 不相连的spans
    "[[1, 2], [4, 5, 6], [5, 6, 7]]",  # 部分相连的spans
    "[[1, 2], [2, 3], [3, 4]]",  # 全部相连的spans
    '[[4]]',
    '[[6,7],[10]]'
]

print("测试结果：")
for test in test_cases:
    result = merge_connected_spans(test)
    print(f"输入: {test}")
    print(f"输出: {result}")
    print("-" * 50)


import difflib
import csv

# 加载数据
final_data = pd.read_csv('1_process_connected_spans_data.csv', sep='\t')  # 诗句和典故的原数据
variation_data = pd.read_csv('典故的异形数据.csv', sep='\t')  # 典故的异形数据

# 使用difflib计算最长公共子串
def longest_common_substring(str1, str2):
    sequence_matcher = difflib.SequenceMatcher(None, str1, str2)
    match = sequence_matcher.find_longest_match(0, len(str1), 0, len(str2))
    return str1[match.a: match.a + match.size]

# 创建一个新DataFrame用于存放修改过的行
modified_rows = []
no_match_rows = []  # 新增：存储没有匹配的行

# 遍历final_data中的每一行
for idx, row in final_data.iterrows():
    if row['allusion_index'] == '[]':
        sentence = row['sentence']
        allusions = row['allusion'].split(';')  # 拆分多个典故
        
        allusion_index = []  # 存储多个典故的匹配位置
        matched_allusions = 0  # 记录匹配成功的典故数

        for allusion in allusions:
            variation_list = variation_data[variation_data['allusion'] == allusion]['variation_list'].values
            
            if len(variation_list) > 0:
                variation_list = eval(variation_list[0])  # 将字符串类型的variation_list转换成列表
                merged_variations = ''.join(variation_list)  # 合并所有变体为一个长字符串
                
                # 查找最长公共子串
                longest_match = longest_common_substring(merged_variations, sentence)
                
                if longest_match:
                    # 更新变体列表，添加新的异形典故
                    if longest_match not in variation_list:
                        variation_list.append(longest_match)

                    # 更新variation_data
                    variation_data.at[variation_data[variation_data['allusion'] == allusion].index[0], 'variation_list'] = str(variation_list)
                    
                    # 记录匹配位置
                    match_start = sentence.find(longest_match)
                    match_end = match_start + len(longest_match)
                    
                    positions = list(range(match_start, match_end + 1))
                    allusion_index.append(positions)  # 记录该典故的所有匹配位置

                    matched_allusions += 1  # 匹配成功的典故数加1

        if matched_allusions > 0:
            final_data.at[idx, 'variation_number'] = matched_allusions  # 更新variation_number为匹配的典故数
            final_data.at[idx, 'allusion_index'] = str(allusion_index)  # 保存多个典故的匹配位置

            # 保存修改后的行
            modified_rows.append(final_data.loc[idx])
        else:
            no_match_rows.append(final_data.loc[idx])  # 如果没有找到匹配的典故，记录此行

# 将修改过的行保存到新的DataFrame
modified_data = pd.DataFrame(modified_rows)

# 将没有匹配的行保存到新的DataFrame
no_match_data = pd.DataFrame(no_match_rows)

# 保存到新的CSV文件
final_data.to_csv('2_processed_data.csv', index=False, sep='\t')
variation_data.to_csv('updated_典故的异性数据.csv', index=False, sep='\t',quoting=csv.QUOTE_NONE)
modified_data.to_csv('modified_rows.csv', index=False, sep='\t')  # 保存修改过的行
no_match_data.to_csv('no_match_rows.csv', index=False, sep='\t')  # 保存没有匹配的行

data = pd.read_csv("2_processed_data.csv",sep = "	")

result = data.groupby('sentence').agg({
    'author': 'first',  # 保留第一条作者
    'title': 'first',   # 保留第一条标题
    'allusion': lambda x: ';'.join(x),  # 合并典故，以分号分隔
    'variation_number': 'sum',  # 求和典故变体数
    'allusion_index': lambda x: ';'.join(x)  # 合并索引，逗号分隔后再用分号分隔不同典故
}).reset_index()

def transform_allusion_index(row):
    # 拆分 allusion 和 allusion_index
    allusions = row['allusion'].split(';')
    allusion_indices = row['allusion_index'].split(';')
    
    # 组合为目标格式
    combined = []
    for allusion, indices in zip(allusions, allusion_indices):
        index_list = indices.strip('[]')  # 去掉索引的外部括号
        combined.append(f"[{index_list},{allusion}]")  # 按目标格式拼接

    return ';'.join(combined)

# 对 DataFrame 的每一行应用函数
result['transformed_allusion'] = result.apply(transform_allusion_index, axis=1)

def adjust_transformed_allusion(row):
    # 获取 transformed_allusion 的值
    transformed_allusion = row['transformed_allusion']
    allusions = row['allusion'].split(';')  # 获取所有典故名列表
    
    # 初始化变量
    adjusted = transformed_allusion
    current_allusion_index = 0  # 当前典故的索引
    
    # 遍历 transformed_allusion 并找到 ] 后的 ,
    i = 0
    while i < len(adjusted):
        if adjusted[i] == ']' and i + 1 < len(adjusted) and adjusted[i + 1] == ',':
            # 在 ] 前插入当前典故名
            current_allusion = allusions[current_allusion_index]
            adjusted = adjusted[:i] + f',{current_allusion}' + adjusted[i:]
            i += len(current_allusion) + 1  # 跳过插入的典故名长度
            
            # 如果当前典故名已经完全应用，切换到下一个典故
            current_allusion_index = min(current_allusion_index + 1, len(allusions) - 1)
        
        i += 1
    
    return adjusted

# 对 DataFrame 的每一行应用函数
result['transformed_allusion'] = result.apply(adjust_transformed_allusion, axis=1)


def replace_comma_after_bracket(row):
    return row['transformed_allusion'].replace('],', '];')

# 对 `transformed_allusion` 列应用替换函数
result['transformed_allusion'] = result.apply(replace_comma_after_bracket, axis=1)

result.to_csv('3_1_1_merged_data.csv', sep='\t', index=False, encoding='utf-8')

def merge_datasets(allusion_file, no_allusion_file, output_file, no_allusion_ratio=0.3):
    """合并包含典故和不包含典故的数据集
    
    Args:
        allusion_file: 包含典故的数据文件路径
        no_allusion_file: 不包含典故的数据文件路径
        output_file: 输出文件路径
        no_allusion_ratio: 不包含典故的样本占总样本的比例
    """
    print("=== 开始合并数据集 ===")
    
    # 读取包含典故的数据
    print(f"读取包含典故的数据: {allusion_file}")
    allusion_df = pd.read_csv(allusion_file, sep='\t')
    
    # 读取不包含典故的数据
    print(f"读取不包含典故的数据: {no_allusion_file}")
    no_allusion_df = pd.read_csv(no_allusion_file, sep='\t')
    
    # 计算需要采样的不包含典故的样本数
    allusion_count = len(allusion_df)
    target_no_allusion_count = int(allusion_count * no_allusion_ratio / (1 - no_allusion_ratio))
    
    print(f"\n数据统计:")
    print(f"包含典故的样本数: {allusion_count}")
    print(f"不包含典故的原始样本数: {len(no_allusion_df)}")
    print(f"目标不包含典故的样本数: {target_no_allusion_count}")
    
    # 随机采样不包含典故的数据
    sampled_no_allusion = no_allusion_df.sample(n=target_no_allusion_count, random_state=42)
    
    # 确保不包含典故的样本的variation_number为0
    sampled_no_allusion['variation_number'] = 0
    sampled_no_allusion['allusion'] = ''
    sampled_no_allusion['allusion_index'] = ''
    
    # 合并数据集
    merged_df = pd.concat([allusion_df, sampled_no_allusion], ignore_index=True)
    
    # 打乱数据顺序
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 保存合并后的数据集
    merged_df.to_csv(output_file, sep='\t', index=False)
    
    print("\n=== 合并完成 ===")
    print(f"最终数据集大小: {len(merged_df)}")
    print(f"包含典故的样本比例: {allusion_count/len(merged_df)*100:.2f}%")
    print(f"不包含典故的样本比例: {len(sampled_no_allusion)/len(merged_df)*100:.2f}%")
    print(f"结果已保存至: {output_file}")
    
    # 打印一些示例
    print("\n数据示例:")
    print("\n包含典故的样本:")
    print(merged_df[merged_df['variation_number'] != 0].head()[['sentence', 'allusion', 'variation_number']].to_string())
    print("\n不包含典故的样本:")
    print(merged_df[merged_df['variation_number'] == 0].head()[['sentence', 'allusion', 'variation_number']].to_string())
    
allusion_file = "3_1_1_merged_data.csv"  # 包含典故的数据
no_allusion_file = "不包含典故.csv"  # 不包含典故的数据
output_file = "3_1_2_final_position_dataset.csv"  # 输出文件

# 合并数据集，不包含典故的样本占30%
merge_datasets(allusion_file, no_allusion_file, output_file, no_allusion_ratio=0.3)

import numpy as np
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

def balance_dataset(input_file, output_file, min_threshold=15, max_ratio=0.35):
    """
    平衡数据集，只对样本数特别少的典故进行补充
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        min_threshold: 最小样本数阈值，低于此值的典故将被平衡
        max_ratio: 最大样本数与平均样本数的比率阈值
    """
    print(f"读取数据文件: {input_file}")
    df = pd.read_csv(input_file, sep='\t')
    
    # 统计原始典故分布
    allusion_counts = Counter(df['allusion'].dropna())
    
    # 计算平均样本数和标准差
    mean_count = np.mean(list(allusion_counts.values()))
    target_count = int(mean_count * (1 + max_ratio))  # 设置目标样本数
    
    print("\n数据统计:")
    print(f"典故总数: {len(allusion_counts)}")
    print(f"平均样本数: {mean_count:.2f}")
    print(f"目标样本数: {target_count}")
    print(f"最小样本数阈值: {min_threshold}")
    
    print("\n原始典故分布:")
    for allusion, count in allusion_counts.most_common():
        print(f"{allusion}: {count}条")
    
    # 准备平衡后的数据
    balanced_data = []
    
    # 对每个典故进行平衡
    for allusion, count in tqdm(allusion_counts.items(), desc="平衡数据"):
        # 获取当前典故的所有样本
        allusion_samples = df[df['allusion'] == allusion]
        balanced_data.append(allusion_samples)  # 保留原始数据
        
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
    print("\n平衡后的典故分布:")
    balanced_counts = Counter(balanced_df['allusion'].dropna())
    for allusion, count in balanced_counts.most_common():
        print(f"{allusion}: {count}条")
    
    # 计算平衡后的统计指标
    balanced_mean = np.mean(list(balanced_counts.values()))
    balanced_std = np.std(list(balanced_counts.values()))
    print(f"\n平衡后统计:")
    print(f"平均样本数: {balanced_mean:.2f}")
    print(f"标准差: {balanced_std:.2f}")
    print(f"原始数据集大小: {len(df)}")
    print(f"平衡后数据集大小: {len(balanced_df)}")
    
    # 保存平衡后的数据集
    balanced_df.to_csv(output_file, sep='\t', index=False)
    print(f"\n平衡后的数据集已保存至: {output_file}")

input_file = "2_processed_data.csv"
output_file = "3_2_balanced_data.csv"
balance_dataset(input_file, output_file)

from sklearn.model_selection import train_test_split
import os

def split_dataset(input_file, dataset_name, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    将数据集分割为训练集、验证集和测试集
    
    Args:
        input_file: 输入文件路径
        dataset_name: 数据集名称（'position' 或 'type'）
        train_ratio: 训练集比例，默认0.8
        val_ratio: 验证集比例，默认0.1
        test_ratio: 测试集比例，默认0.1
        random_state: 随机种子，默认42
    """
    print(f"开始读取数据文件: {input_file}")
    
    # 读取CSV文件
    df = pd.read_csv(input_file, sep='\t')
    print(f"总数据量: {len(df)}")
    
    # 获取包含典故的样本
    df_with_allusions = df[df['variation_number'] != '0']
    print(f"包含典故的样本数: {len(df_with_allusions)}")
    
    # 首先分出测试集
    train_val_df, test_df = train_test_split(
        df_with_allusions,
        test_size=test_ratio,
        random_state=random_state,
        shuffle=True
    )
    
    # 从剩余数据中分出验证集
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        shuffle=True
    )
    
    # 打乱数据顺序
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # 保存文件
    train_file = f'4_train_{dataset_name}.csv'
    val_file = f'4_val_{dataset_name}.csv'
    test_file = f'4_test_{dataset_name}.csv'
    
    train_df.to_csv(train_file, sep='\t', index=False)
    val_df.to_csv(val_file, sep='\t', index=False)
    test_df.to_csv(test_file, sep='\t', index=False)
    
    print("\n数据集分割完成:")
    print(f"训练集大小: {len(train_df)} ({len(train_df)/len(df_with_allusions)*100:.1f}%)")
    print(f"验证集大小: {len(val_df)} ({len(val_df)/len(df_with_allusions)*100:.1f}%)")
    print(f"测试集大小: {len(test_df)} ({len(test_df)/len(df_with_allusions)*100:.1f}%)")
    print(f"\n文件保存位置:")
    print(f"训练集: {train_file}")
    print(f"验证集: {val_file}")
    print(f"测试集: {test_file}")


input_file = "3_1_2_final_position_dataset.csv"
split_dataset(input_file, 'position')

input_file = "3_2_balanced_data.csv"
split_dataset(input_file, 'type')