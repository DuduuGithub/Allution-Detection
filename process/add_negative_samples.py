import pandas as pd
import os
from sklearn.model_selection import train_test_split

def add_negative_samples(no_allusion_file, train_file, val_file, test_file):
    """
    将不包含典故的样本按8:1:1的比例添加到训练、验证和测试集中
    
    Args:
        no_allusion_file: 不包含典故样本的文件路径
        train_file: 训练集文件路径
        val_file: 验证集文件路径
        test_file: 测试集文件路径
    """
    print(f"读取不包含典故样本: {no_allusion_file}")
    no_allusion_df = pd.read_csv(no_allusion_file, sep='\t')
    
    # 按8:1:1的比例分割不包含典故的样本
    train_neg, temp = train_test_split(no_allusion_df, train_size=0.8, random_state=42)
    val_neg, test_neg = train_test_split(temp, test_size=0.5, random_state=42)
    
    # 读取原始数据集
    print("读取原始数据集...")
    train_df = pd.read_csv(train_file, sep='\t')
    val_df = pd.read_csv(val_file, sep='\t')
    test_df = pd.read_csv(test_file, sep='\t')
    
    # 合并数据集
    train_merged = pd.concat([train_df, train_neg], ignore_index=True)
    val_merged = pd.concat([val_df, val_neg], ignore_index=True)
    test_merged = pd.concat([test_df, test_neg], ignore_index=True)
    
    # 打乱数据顺序
    train_merged = train_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    val_merged = val_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    test_merged = test_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 生成新文件名
    train_output = train_file.replace('no_bug', 'no_bug_with_neg')
    val_output = val_file.replace('no_bug', 'no_bug_with_neg')
    test_output = test_file.replace('no_bug', 'no_bug_with_neg')
    
    # 保存结果
    train_merged.to_csv(train_output, sep='\t', index=False)
    val_merged.to_csv(val_output, sep='\t', index=False)
    test_merged.to_csv(test_output, sep='\t', index=False)
    
    # 打印统计信息
    print("\n处理完成:")
    print(f"训练集:")
    print(f"  - 原始样本数: {len(train_df)}")
    print(f"  - 添加负样本数: {len(train_neg)}")
    print(f"  - 最终样本数: {len(train_merged)}")
    
    print(f"\n验证集:")
    print(f"  - 原始样本数: {len(val_df)}")
    print(f"  - 添加负样本数: {len(val_neg)}")
    print(f"  - 最终样本数: {len(val_merged)}")
    
    print(f"\n测试集:")
    print(f"  - 原始样本数: {len(test_df)}")
    print(f"  - 添加负样本数: {len(test_neg)}")
    print(f"  - 最终样本数: {len(test_merged)}")
    
    print("\n保存路径:")
    print(f"训练集: {train_output}")
    print(f"验证集: {val_output}")
    print(f"测试集: {test_output}")

if __name__ == "__main__":
    no_allusion_file = "data/used_no_allusion_samples.csv"
    train_file = "data/4_train_type_no_bug.csv"
    val_file = "data/4_val_type_no_bug.csv"
    test_file = "data/4_test_type_no_bug.csv"
    
    add_negative_samples(no_allusion_file, train_file, val_file, test_file) 