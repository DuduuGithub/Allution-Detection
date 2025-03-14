import pandas as pd

def del_much_negatives(file_path, output_path):
    # 读取数据
    df = pd.read_csv(file_path, delimiter="\t")  # 如果是Tab分隔，请指定 `\t`

    # 统计 variation_number 为 0 的数据
    zero_variation_df = df[df["variation_number"] == 0]
    non_zero_variation_df = df[df["variation_number"] != 0]

    # 计算 80% 的数量
    drop_count = int(len(zero_variation_df) * 0.8)

    # 随机采样 80% 的 variation_number 为 0 的数据
    drop_indices = zero_variation_df.sample(n=drop_count, random_state=42).index

    # 删除采样的行
    filtered_df = df.drop(index=drop_indices)

    # 保存处理后的数据
    filtered_df.to_csv(output_path, index=False, sep="\t")  # 仍使用Tab分隔

    print(f"原始数据大小: {df.shape[0]} 行")
    print(f"删除了 {drop_count} 行 variation_number 为 0 的数据")
    print(f"处理后数据大小: {filtered_df.shape[0]} 行")

if __name__ == "__main__":
    del_much_negatives("data/4_train_position_no_bug.csv", "data/4_train_position_no_bug_less_negatives.csv")
    del_much_negatives("data/4_test_position_no_bug.csv", "data/4_test_position_no_bug_less_negatives.csv")
    del_much_negatives("data/4_val_position_no_bug.csv", "data/4_val_position_no_bug_less_negatives.csv")
