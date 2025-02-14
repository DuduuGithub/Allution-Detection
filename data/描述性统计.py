import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取CSV文件
df = pd.read_csv('data/爬取的典故数据.csv', sep='\t')

# 统计每个典故的出现次数
allusion_counts = df['allusion'].value_counts()

# 设置区间
bins = [10, 20, 30, 40, 50, 60,70, 80, 90, 100, float('inf')]
labels = ['10-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '100+']

# 统计各区间的典故数量
hist_data = pd.cut(allusion_counts.values, bins=bins, labels=labels).value_counts().sort_index()

# 创建图形
plt.figure(figsize=(12, 6))
sns.barplot(x=hist_data.index, y=hist_data.values)

# 设置标题和标签
plt.title('典故使用频率分布', fontsize=14)
plt.xlabel('出现次数区间', fontsize=12)
plt.ylabel('典故数量', fontsize=12)

# 在每个柱子上添加数值标签
for i, v in enumerate(hist_data.values):
    plt.text(i, v, f' {v}', ha='center', va='bottom')

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('典故频率分布统计.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()

# 计算统计指标
mean_count = allusion_counts.mean()
median_count = allusion_counts.median()
mode_count = allusion_counts.mode()[0]
total_allusions = len(allusion_counts)
total_poems = len(df)

# 打印统计结果
print("\n统计分析结果：")
print(f"典故总数：{total_allusions}")
print(f"诗句总数：{total_poems}")
print(f"每个典故平均出现次数：{mean_count:.2f}")
print(f"典故出现次数中位数：{median_count}")
print(f"典故出现次数众数：{mode_count}")

print("\n各区间典故数量分布：")
for interval, count in hist_data.items():
    print(f"{interval}次: {count}个典故")

# 找出出现次数最多的前5个典故
print("\n出现次数最多的典故：")
top_5 = allusion_counts.head()
for allusion, count in top_5.items():
    print(f"{allusion}: {count}次")

# 保存统计结果到文件
with open('典故统计结果.txt', 'w', encoding='utf-8') as f:
    f.write("统计分析结果：\n")
    f.write(f"典故总数：{total_allusions}\n")
    f.write(f"诗句总数：{total_poems}\n")
    f.write(f"每个典故平均出现次数：{mean_count:.2f}\n")
    f.write(f"典故出现次数中位数：{median_count}\n")
    f.write(f"典故出现次数众数：{mode_count}\n")
    
    f.write("\n各区间典故数量分布：\n")
    for interval, count in hist_data.items():
        f.write(f"{interval}次: {count}个典故\n")
    
    f.write("\n出现次数最多的典故：\n")
    for allusion, count in top_5.items():
        f.write(f"{allusion}: {count}次\n")

print("统计结果已保存到 '典故统计结果.txt'")