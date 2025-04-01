import pandas as pd

def split_multiple_allusions(df):
    # 创建新的行列表
    new_rows = []
    
    for _, row in df.iterrows():
        # 如果没有典故 (transformed_allusion为空)
        if pd.isna(row['transformed_allusion']):
            new_rows.append(row)
            continue
        
        # 检查是否有多个典故 (通过分号分隔)
        if ';' in row['transformed_allusion']:
            # 分割转换后的位置
            transformed = row['transformed_allusion'].split(';')
            # 为每个典故创建新行
            for t in transformed:
                new_row = row.copy()  # 每个典故创建新的行副本
                new_row['singling_type'] = t.strip()
                new_rows.append(new_row)
        else:
            # 单个典故的情况
            new_row = row.copy()
            new_rows.append(new_row)
            
    # 创建新的DataFrame
    new_df = pd.DataFrame(new_rows)
    return new_df

def process_file(input_path, output_path):
    # 读取CSV文件
    df = pd.read_csv(input_path, sep='\t')
    
    # 处理多典故
    new_df = split_multiple_allusions(df)
    
    # 保存处理后的文件
    new_df.to_csv(output_path, sep='\t', index=False)
    
# 处理所有文件
file_types = ['train', 'val', 'test']
for file_type in file_types:
    input_path = f'data/4_{file_type}_position_no_bug_less_negatives.csv'
    output_path = f'data/5_{file_type}_position_no_bug_less_negatives_single_allusion.csv'
    process_file(input_path, output_path)