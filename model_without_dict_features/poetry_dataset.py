import torch
from torch.utils.data import Dataset
import random  # 添加在文件开头的import部分


class PoetryNERDataset(Dataset):
    '''
        在定义时执行__init__(),运行read_data()读取到数据集中的数据
        在for batch in dataloader中，对batch的每一个item，执行__getitem__()，返回一个batch的数据,再执行collate_fn()，返回一个batch的数据
    '''
    def __init__(self, file_path, tokenizer, max_len, type_label2id,negative_sample_ratio=0.01):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.type_label2id = type_label2id
        
        # 典故位置标注标签映射
        self.position_label2id = {
            'O': 0,
            'B': 1,
            'I': 2
        }
            
        self.data = self.read_data(file_path)
        self.negative_sample_ratio = negative_sample_ratio  # 添加负采样比例参数
    
    def read_data(self, file_path):
        """读取数据文件"""
        dataset = []
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # 跳过标题行
            for line in f:
                if not line.strip() or '\t' not in line:
                    continue
                
                result = self.parse_line(line)
                if result is None:
                    continue
                
                text, position_labels, allusion_info = result
                dataset.append({
                    'text': text,
                    'position_labels': position_labels,
                    'allusion_info': allusion_info
                })
        
        return dataset

    def parse_line(self, line):
        """解析单行数据，同时提取位置和类型标签"""
        parts = line.strip().split('\t')
        try:
            text = parts[0].strip()
            variation_number = int(parts[4])
            
            # 初始化标签
            position_labels = ['O'] * len(text)
            allusion_info = []  # [(start_pos, end_pos, type_label), ...]
            
            if variation_number > 0:
                # 处理有典故的情况
                transformed_allusions = parts[6].strip().split(';')
                for allusion in transformed_allusions:
                    # 去除最外层的方括号
                    allusion = allusion.strip('[] \t\n\r')
                    if not allusion:
                        continue
                        
                    # 分离位置和类型
                    pos_end = allusion.rfind(',')
                    if pos_end == -1:
                        continue
                        
                    positions_str = allusion[:pos_end].strip('[]')
                    # 提取并清理类型标签，处理括号内的内容
                    allusion_type = allusion[pos_end + 1:].strip()
                    
                    try:
                        # 解析位置字符串为数字列表
                        positions = [int(pos.strip()) for pos in positions_str.split(',')]
                        
                        if positions:
                            # 设置位置标签
                            position_labels[positions[0]] = 'B'
                            for pos in positions[1:]:
                                position_labels[pos] = 'I'
                            
                            # 记录典故信息
                            allusion_info.append((positions[0], positions[-1], allusion_type))
                    except ValueError:
                        print(f"Warning: Invalid position format in {allusion}")
                        continue
            
            # 转换标签为ID
            position_ids = [self.position_label2id[label] for label in position_labels]
            
            return text, position_ids, allusion_info
            
        except (IndexError, ValueError) as e:
            print(f"Error parsing line: {line}")
            print(f"Error details: {str(e)}")
            return None

    def __getitem__(self, idx):
        '''
        返回一个item的数据：
            text 原始文本，没有 CLS/SEP
            input_ids      有 CLS/SEP
            attention_mask 记录 有 CLS/SEP
                     # [CLS]  今    日    江    南   [SEP]  [PAD]  [PAD]
            mask:    # [  1    1     1     1     1     1      0      0  ]
            position_labels 有 CLS/SEP
            target_positions 有 CLS/SEP
            type_labels 没有 CLS/SEP
            dict_features 有 CLS/SEP
        '''
        item = self.data[idx]
        text = item['text']
        
        # BERT tokenization
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 准备基础特征
        result = {
            'text': text,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        
        # 添加位置标签
        position_labels = item['position_labels']
        padded_position_labels = torch.zeros(len(result['input_ids']), dtype=torch.long)
        padded_position_labels[1:len(text)+1] = torch.tensor(position_labels)
        result['position_labels'] = padded_position_labels
        
        # 添加类型标签信息
        text_len = len(text)
        if item['allusion_info']:
            type_labels = []
            target_positions = []
            
            for start, end, type_label in item['allusion_info']:
                # 随机决定是否将正例样本转为负例
                if random.random() < self.negative_sample_ratio:
                    # 计算实际文本长度（不包括CLS、SEP和padding）
                    actual_text_len = len(text)
                    
                    # 随机选择一个不包含典故的位置，只在实际文本范围内选择
                    valid_positions = []
                    for i in range(actual_text_len):
                        # 确保结束位置不超过实际文本长度
                        possible_end = min(i + end - start, actual_text_len - 1)
                        # 检查这个范围是否与任何已知典故重叠
                        if not any(s <= i <= e or s <= possible_end <= e 
                                  for s, e, _ in item['allusion_info']):
                                valid_positions.append((i, possible_end))
                    
                    if valid_positions:
                        new_start, new_end = random.choice(valid_positions)
                        # +1 是因为要考虑 [CLS]，但确保不会超出实际长度
                        target_positions.append([new_start + 1, new_end + 1])
                        type_labels.append(self.type_label2id['O'])
                    else:
                        # 如果找不到合适的负例位置，保持原来的正例
                        target_positions.append([start + 1, end + 1])
                        type_labels.append(self.type_label2id[type_label])
                else:
                    target_positions.append([start + 1, end + 1])
                    type_labels.append(self.type_label2id[type_label])
        else:
            # 对于不包含典故的样本，随机选择一个位置
            start = random.randint(0, text_len-2)  # 至少选择2个字符
            end = min(start + random.randint(1, 3), text_len-1)  # 随机长度1-3
            target_positions = [[start + 1, end + 1]]  # +1 因为[CLS]
            type_labels = [self.type_label2id['O']]
        
        result['target_positions'] = torch.tensor(target_positions)
        result['type_labels'] = torch.tensor(type_labels)
        
        return result

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        """自定义批处理函数，处理不同长度的特征"""
        # 获取这个batch中的最大文本长度
        max_text_len = max(len(item['text']) for item in batch)
        max_seq_len = max(len(item['input_ids']) for item in batch)
        
        # 获取这个batch中type_labels的最大长度
        max_type_len = max(len(item['type_labels']) for item in batch)
        
        # 准备batch数据
        batch_texts = []
        batch_input_ids = []
        batch_attention_mask = []
        batch_position_labels = []
        batch_type_labels = []
        batch_target_positions = []
        
        for item in batch:
            text = item['text']
            batch_texts.append(text)
            
            # 填充 input_ids 和 attention_mask 到最大长度
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            
            if len(input_ids) < max_seq_len:
                # 填充 input_ids
                padding = torch.zeros(max_seq_len - len(input_ids), dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, padding])
                # 填充 attention_mask
                padding = torch.zeros(max_seq_len - len(attention_mask), dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, padding])
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            
            # 处理位置标签
            position_labels = item['position_labels']
            padded_position_labels = torch.zeros(max_seq_len, dtype=torch.long)
            padded_position_labels[1:len(text)+1] = position_labels[1:len(text)+1]  # 跳过[CLS]
            batch_position_labels.append(padded_position_labels)
            
            # 处理类型标签，填充到最大长度
            type_labels = item['type_labels']
            padded_type_labels = torch.zeros(max_type_len, dtype=torch.long)
            padded_type_labels[:len(type_labels)] = type_labels
            batch_type_labels.append(padded_type_labels)
            
            # 处理目标位置，确保维度一致
            target_positions = item['target_positions']
            if len(target_positions) < max_type_len:
                padding = torch.zeros((max_type_len - len(target_positions), 2), dtype=torch.long)
                target_positions = torch.cat([target_positions, padding])
            batch_target_positions.append(target_positions)
        
        # 构建返回字典
        return {
            'text': batch_texts,
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
            'position_labels': torch.stack(batch_position_labels),
            'type_labels': torch.stack(batch_type_labels),
            'target_positions': torch.stack(batch_target_positions)
        }
