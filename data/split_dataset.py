import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(input_file, train_ratio=0.8, random_state=42):
    """
    将数据集分割为训练集和测试集
    
    Args:
        input_file: 输入文件路径 (final_data.csv)
        train_ratio: 训练集比例，默认0.8
        random_state: 随机种子，默认42
    """
    print(f"开始读取数据文件: {input_file}")
    
    # 读取CSV文件
    df = pd.read_csv(input_file, sep='\t')
    print(f"总数据量: {len(df)}")
    
    # 获取包含典故的样本
    df_with_allusions = df[df['variation_number'] != '0']
    
    print(f"包含典故的样本数: {len(df_with_allusions)}")
    
    # 分别对包含典故和不包含典故的样本进行分割
    train_df, test_df = train_test_split(
        df_with_allusions,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True
    )
    
    # 打乱数据顺序
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # 创建保存目录
    os.makedirs('data', exist_ok=True)
    
    # 保存文件
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'
    
    train_df.to_csv(train_file, sep='\t', index=False)
    test_df.to_csv(test_file, sep='\t', index=False)
    
    print("\n数据集分割完成:")
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"训练集保存至: {train_file}")
    print(f"测试集保存至: {test_file}")
    
    # 打印包含典故的样本统计
    print("\n包含典故的样本统计:")
    print(f"训练集中包含典故的样本数: {len(train_df[train_df['variation_number'] != '0'])}")
    print(f"测试集中包含典故的样本数: {len(test_df[test_df['variation_number'] != '0'])}")

if __name__ == "__main__":
    input_file = "data/final_data.csv"
    split_dataset(input_file) 