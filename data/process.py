import pandas as pd
import ast
import numpy as np

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
        if not isinstance(span, list) or len(span) < 2:
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
df = pd.read_csv('data/爬取的典故数据.csv', sep='\t')

# 处理allusion_index列
df['allusion_index'] = df['allusion_index'].apply(merge_connected_spans)

# 保存处理后的数据
df.to_csv('data/processed_典故数据.csv', sep='\t', index=False)

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
]

print("测试结果：")
for test in test_cases:
    result = merge_connected_spans(test)
    print(f"输入: {test}")
    print(f"输出: {result}")
    print("-" * 50)
