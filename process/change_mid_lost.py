import pandas as pd
import ast
import os

def expand_positions_with_allusion_position(position_str):
    """
    将transformed_allusion中的位置扩展为包含所有中间位置
    例如: "[6, 9,五柳（陶潜）]" -> "[6, 7, 8, 9,五柳（陶潜）]"
    """
    try:
        # 分割多个典故（以;分隔）
        allusions = position_str.split(';')
        expanded_allusions = []
        
        for allusion in allusions:
            # 去除方括号并分割
            content = allusion.strip('[]').split(',')
            
            # 提取位置和典故名称
            positions = [int(content[0]), int(content[1])]
            allusion_name = ','.join(content[2:])  # 处理典故名称中可能包含逗号的情况
            
            # 展开位置
            expanded_positions = list(range(positions[0], positions[1] + 1))
            
            # 组合展开的位置和典故名称
            expanded_allusion = '[' + ', '.join(map(str, expanded_positions)) + ',' + allusion_name + ']'
            expanded_allusions.append(expanded_allusion)
        
        # 用分号连接多个典故
        return ';'.join(expanded_allusions)
    except:
        # 如果解析失败，返回原字符串
        return position_str

def expand_positions_with_allusion_type(position_str):
    """
    将区间标注展开为完整序列
    例如: [[8, 10]] -> [[8, 9, 10]]
         [[0, 2], [4, 5]] -> [[0,1,2], [4,5]]
    """
    try:
        # 将字符串转换为Python列表
        ranges = ast.literal_eval(position_str)
        
        # 如果是空列表，直接返回
        if not ranges:
            return str([])
        
        expanded_ranges = []
        for range_pair in ranges:
            start, end = range_pair
            # 展开每个子区间
            expanded_range = list(range(start, end + 1))
            expanded_ranges.append(expanded_range)
        
        return str(expanded_ranges)
    except:
        # 如果解析失败，返回原字符串
        return position_str


def process_file(input_file, output_file,task):
    """
    处理CSV文件，展开transformed_allusion中的位置标记
    """
    # 读取CSV文件
    df = pd.read_csv(input_file, sep='\t')  # 使用制表符分隔
    
    # 应用转换到transformed_allusion列
    if task == 'position':
        df['transformed_allusion'] = df['transformed_allusion'].apply(expand_positions_with_allusion_position)
    elif task == 'type':
        df['allusion_index'] = df['allusion_index'].apply(expand_positions_with_allusion_type)
    
    # 保存结果，使用制表符分隔
    df.to_csv(output_file, sep='\t', index=False)
    print(f"处理完成，结果已保存到: {output_file}")

def main():
    # 设置输入输出文件路径
    # input_file_1 = 'data/4_train_position.csv'
    # input_file_2 = 'data/4_val_position.csv'
    # input_file_3 = 'data/4_test_position.csv'
    
    input_file_4 = 'data/4_train_type.csv'
    input_file_5 = 'data/4_val_type.csv'
    input_file_6 = 'data/4_test_type.csv'
    
    
    # output_file_1 = 'data/4_train_position_no_bug.csv'
    # output_file_2 = 'data/4_val_position_no_bug.csv'
    # output_file_3 = 'data/4_test_position_no_bug.csv'
    
    output_file_4 = 'data/4_train_type_no_bug.csv'
    output_file_5 = 'data/4_val_type_no_bug.csv'
    output_file_6 = 'data/4_test_type_no_bug.csv'

    # 处理文件
    # process_file(input_file_1, output_file_1,'position')
    # process_file(input_file_2, output_file_2,'position')
    # process_file(input_file_3, output_file_3,'position')
    process_file(input_file_4, output_file_4,'type')
    process_file(input_file_5, output_file_5,'type')
    process_file(input_file_6, output_file_6,'type')
if __name__ == "__main__":
    main()
