import pandas as pd

def load_reference_allusions(reference_file):
    # 读取参考典故数据
    ref_df = pd.read_csv(reference_file, sep='\t')
    # 创建一个包含所有合法典故的集合
    valid_allusions = set()
    for allusion in ref_df['allusion']:
        # 处理可能包含分号的情况
        parts = str(allusion).split(';')
        valid_allusions.update(part.strip() for part in parts)
    return valid_allusions

def check_and_remove_invalid_allusions(file_path, valid_allusions):
    # 读取数据文件
    df = pd.read_csv(file_path, sep='\t')
    
    def is_valid_allusion(allusion_str):
        # 检查典故是否在有效典故集合中
        parts = str(allusion_str).split(';')
        return all(part.strip() in valid_allusions for part in parts)
    
    # 创建过滤条件
    mask = df['allusion'].apply(is_valid_allusion)
    
    # 找出被删除的典故
    removed_df = df[~mask]
    if len(removed_df) > 0:
        print(f"\n在 {file_path} 中发现以下无效典故：")
        for idx, row in removed_df.iterrows():
            print(f"- {row['allusion']}")
    
    # 应用过滤条件
    filtered_df = df[mask]
    
    # 保存结果到原文件
    filtered_df.to_csv(file_path, sep='\t', index=False)
    
    # 返回删除的记录数
    removed_count = len(df) - len(filtered_df)
    return removed_count

if __name__ == "__main__":
    # 文件路径
    reference_file = "data_for_new_study/variant_allusion_data.csv"
    file_paths = [
        "data_for_new_study/filtered_data.csv",
        "data_for_new_study/train_data.csv",
        "data_for_new_study/valid_data.csv",
        "data_for_new_study/test_data.csv"
    ]
    
    # 加载参考典故集合
    valid_allusions = load_reference_allusions(reference_file)
    
    # 处理每个数据文件
    for file_path in file_paths:
        removed_count = check_and_remove_invalid_allusions(file_path, valid_allusions)
        print(f"从 {file_path} 中删除了 {removed_count} 条记录")