import pandas as pd

def extract_no_allusion_samples(input_file, output_file):
    """
    从数据集中提取不包含典故的样本
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"读取数据文件: {input_file}")
    
    # 读取CSV文件
    df = pd.read_csv(input_file, sep='\t')
    
    # 筛选不包含典故的样本（variation_number为'0'的样本）
    no_allusion_df = df[df['variation_number'] == 0]
    
    # 保存结果
    no_allusion_df.to_csv(output_file, sep='\t', index=False)
    
    print(f"\n处理完成:")
    print(f"原始数据总量: {len(df)}")
    print(f"不包含典故的样本数: {len(no_allusion_df)}")
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    input_file = "data/3_1_2_final_position_dataset.csv"
    output_file = "data/used_no_allusion_samples.csv"
    extract_no_allusion_samples(input_file, output_file) 