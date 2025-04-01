import pandas as pd
import os

def remove_long_sentences(file_path, max_length=120):
    # 读取数据文件
    df = pd.read_csv(file_path, sep='\t')
    
    # 获取原始行数
    original_count = len(df)
    
    # 找出需要删除的句子
    long_sentences = df[df['sentence'].str.len() > max_length]
    
    if len(long_sentences) > 0:
        print(f"\n在 {file_path} 中发现以下过长句子：")
        for idx, row in long_sentences.iterrows():
            print(f"- 长度 {len(row['sentence'])}: {row['sentence']}")
    
    # 过滤掉长句子
    df = df[df['sentence'].str.len() <= max_length]
    
    # 保存结果
    df.to_csv(file_path, sep='\t', index=False)
    
    # 返回删除的数量
    removed_count = original_count - len(df)
    return removed_count

def main():
    # 指定要处理的文件路径
    file_paths = [
        'data_for_new_study/filtered_data.csv',
        'data_for_new_study/train_data.csv',
        'data_for_new_study/test_data.csv',
        'data_for_new_study/val_data.csv'
    ]
    
    total_removed = 0
    
    # 处理每个文件
    for file_path in file_paths:
        if os.path.exists(file_path):
            removed_count = remove_long_sentences(file_path)
            print(f"从 {file_path} 中删除了 {removed_count} 条过长记录")
            total_removed += removed_count
        else:
            print(f"文件不存在: {file_path}")
    
    print(f"\n总共删除了 {total_removed} 条过长记录")

if __name__ == "__main__":
    main() 