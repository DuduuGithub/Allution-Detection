

# 加载数据
final_data = pd.read_csv('1_process_connected_spans_data.csv', sep='\t')  # 诗句和典故的原数据
variation_data = pd.read_csv('典故的异形数据.csv', sep='\t')  # 典故的异形数据

# 使用difflib计算最长公共子串
def longest_common_substring(str1, str2):
    sequence_matcher = difflib.SequenceMatcher(None, str1, str2)
    match = sequence_matcher.find_longest_match(0, len(str1), 0, len(str2))
    return str1[match.a: match.a + match.size]

# 创建一个新DataFrame用于存放修改过的行
modified_rows = []
no_match_rows = []  # 新增：存储没有匹配的行

# 遍历final_data中的每一行
for idx, row in final_data.iterrows():
    if row['allusion_index'] == '[]':
        sentence = row['sentence']
        allusions = row['allusion'].split(';')  # 拆分多个典故
        
        allusion_index = []  # 存储多个典故的匹配位置
        matched_allusions = 0  # 记录匹配成功的典故数

        for allusion in allusions:
            variation_list = variation_data[variation_data['allusion'] == allusion]['variation_list'].values
            
            if len(variation_list) > 0:
                variation_list = eval(variation_list[0])  # 将字符串类型的variation_list转换成列表
                merged_variations = ''.join(variation_list)  # 合并所有变体为一个长字符串
                
                # 查找最长公共子串
                longest_match = longest_common_substring(merged_variations, sentence)
                
                if longest_match:
                    # 更新变体列表，添加新的异形典故
                    if longest_match not in variation_list:
                        variation_list.append(longest_match)

                    # 更新variation_data
                    variation_data.at[variation_data[variation_data['allusion'] == allusion].index[0], 'variation_list'] = str(variation_list)
                    
                    # 记录匹配位置
                    match_start = sentence.find(longest_match)
                    match_end = match_start + len(longest_match)
                    
                    positions = list(range(match_start, match_end + 1))
                    allusion_index.append(positions)  # 记录该典故的所有匹配位置

                    matched_allusions += 1  # 匹配成功的典故数加1

        if matched_allusions > 0:
            final_data.at[idx, 'variation_number'] = matched_allusions  # 更新variation_number为匹配的典故数
            final_data.at[idx, 'allusion_index'] = str(allusion_index)  # 保存多个典故的匹配位置

            # 保存修改后的行
            modified_rows.append(final_data.loc[idx])
        else:
            no_match_rows.append(final_data.loc[idx])  # 如果没有找到匹配的典故，记录此行

# 将修改过的行保存到新的DataFrame
modified_data = pd.DataFrame(modified_rows)

# 将没有匹配的行保存到新的DataFrame
no_match_data = pd.DataFrame(no_match_rows)

# 保存到新的CSV文件
final_data.to_csv('2_processed_data.csv', index=False, sep='\t')
variation_data.to_csv('updated_典故的异性数据.csv', index=False, sep='\t',quoting=csv.QUOTE_NONE)
modified_data.to_csv('modified_rows.csv', index=False, sep='\t')  # 保存修改过的行
no_match_data.to_csv('no_match_rows.csv', index=False, sep='\t')  # 保存没有匹配的行