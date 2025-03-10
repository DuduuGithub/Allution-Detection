import torch
from torch.utils.data import Dataset
import re
from transformers import BertTokenizer
import csv
from torch.utils.data import DataLoader
    
class PoetryNERDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len,type_label2id,id2type_label, task='position'):
        """
        Args:
            file_path: 数据文件路径
            tokenizer: BERT tokenizer
            max_len: 最大序列长度
            task: 'position' 用于典故位置识别, 'type' 用于典故类型分类
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task = task
        
        # 典故位置标注标签映射
        self.position_label2id = {
            'O': 0,
            'B': 1,
            'I': 2
        }
        
        # 从固定文件加载类型映射
        self.type_label2id=type_label2id
        self.id2type_label=id2type_label
        
        # 加载数据
        self.data = self.read_data(file_path)
    
    
    
    def parse_line(self, line):
        """解析单行数据，提取诗句和标签"""
        parts = line.strip().split('\t')
        try:
            # 获取基本信息
            text = parts[0].strip()
            variation_number = int(parts[4])
            
            # 如果没有典故（variation_number为0），直接返回全O标签
            if variation_number == 0:
                position_labels = [self.position_label2id['O']] * len(text)
                type_labels = [-1] * len(text)
                return text, position_labels, type_labels
            
            # 处理有典故的情况
            allusion_info = parts[6].strip()
            allusion_parts = allusion_info.split(';')
            
            # 初始化标签序列
            position_labels = ['O'] * len(text)
            type_labels = ['O'] * len(text) # 'O'表示非典故
            
            # 处理每个典故
            for part in allusion_parts:
                part = part.strip()
                if not part:  # 跳过空字符串
                    continue
                
                part = part.strip('[]')
                items = [item.strip() for item in part.split(',')]
                positions = [int(pos) for pos in items[:-1]]
                allusion_type = items[-1]
                
                # 设置位置标签
                if positions:
                    position_labels[positions[0]] = 'B'
                    for pos in positions[1:]:
                        position_labels[pos] = 'I'
                    
                    # 设置类型标签
                    for pos in positions:
                        type_labels[pos] = allusion_type
            
            # 将标签转换为ID
            position_ids = [self.position_label2id[label] for label in position_labels]
            type_ids = [self.type_label2id.get(label, -1) for label in type_labels]
            
            return text, position_ids, type_ids
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line}")
            print(f"Error details: {str(e)}")
            return None
        
    
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
                
                text, position_labels, type_labels = result # 这些都是标号
                
                if self.task == 'position':
                    dataset.append({
                        'text': text,
                        'position_labels': position_labels,
                    })
                else:  # task == 'type'
                    # 为每个典故创建一个单独的样本
                    allusion_positions = self.get_allusion_positions(position_labels, type_labels)
                    for start, end, type_label in allusion_positions:
                        dataset.append({
                            'text': text,
                            'position_labels': position_labels,
                            'type_labels': type_labels,
                            'target_positions': (start, end),
                            'target_type': type_label
                        })
        
        return dataset

    def get_allusion_positions(self, position_labels, type_labels):
        """提取所有典故的位置和类型"""
        allusion_positions = []
        i = 0
        while i < len(position_labels):
            if position_labels[i] == self.position_label2id['B']:
                start = i
                end = i
                # 寻找典故结束位置
                for j in range(i + 1, len(position_labels)):
                    if position_labels[j] == self.position_label2id['I']:
                        end = j
                    else:
                        break
                allusion_positions.append((start, end, type_labels[start]))
                i = end + 1
            else:
                i += 1
        return allusion_positions

    def __getitem__(self, idx):
        """获取单个样本"""
        text = self.data[idx]['text']
        position_labels = self.data[idx]['position_labels']
        
        # BERT tokenization
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  # 添加[CLS]和[SEP]
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        print('text:',text)
        # 为[CLS]和[SEP]准备标签
        padded_position_labels = torch.zeros(len(input_ids), dtype=torch.long)
        padded_position_labels[1:len(text)+1] = torch.tensor(position_labels)  # 跳过[CLS]
        
        if self.task == 'type':
            type_labels = self.data[idx]['type_labels']
            padded_type_labels = torch.zeros(len(input_ids), dtype=torch.long)
            padded_type_labels[1:len(text)+1] = torch.tensor(type_labels)  # 跳过[CLS]
            
            # 目标位置也需要考虑[CLS]的偏移
            target_positions = self.data[idx]['target_positions']
            adjusted_positions = [pos + 1 for pos in target_positions]  # 位置+1以适应[CLS]
            
            return {
                'text': text,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'type_labels': padded_type_labels,
                'target_positions': torch.tensor(adjusted_positions)
            }
        
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_labels': padded_position_labels
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        """自定义批处理函数，动态调整每个batch的长度"""
        # 获取这个batch中的最大文本长度
        max_text_len = max(len(item['text']) for item in batch)
        max_seq_len = max_text_len + 2  # 加2是为了CLS和SEP
        
        # 准备batch数据
        batch_texts = []
        batch_input_ids = []
        batch_attention_mask = []
        batch_position_labels = []
        
        for item in batch:
            text = item['text']
            
            # 重新进行tokenization，使用当前batch的最大长度
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                padding='max_length',
                max_length=max_seq_len,
                truncation=True,
                return_tensors='pt'
            )
            
            batch_texts.append(text)
            batch_input_ids.append(encoding['input_ids'].squeeze(0))
            batch_attention_mask.append(encoding['attention_mask'].squeeze(0))
            
            # 处理标签
            if self.task == 'position':
                position_labels = item['position_labels']
                padded_labels = torch.zeros(max_seq_len, dtype=torch.long)
                padded_labels[1:len(text)+1] = position_labels[1:len(text)+1]  # 保持CLS的标签为0
                batch_position_labels.append(padded_labels)
        
        # 将列表转换为张量
        batch_dict = {
            'text': batch_texts,
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
        }
        
        if self.task == 'position':
            batch_dict['position_labels'] = torch.stack(batch_position_labels)
        else:  # task == 'type'
            batch_type_labels = []
            batch_target_positions = []
            
            for item in batch:
                # 类型标签和目标位置已经在__getitem__中考虑了CLS的偏移
                type_labels = item['type_labels']
                padded_type_labels = torch.zeros(max_seq_len, dtype=torch.long)
                padded_type_labels[1:len(text)+1] = type_labels[1:len(text)+1]
                batch_type_labels.append(padded_type_labels)
                batch_target_positions.append(item['target_positions'])
            
            batch_dict['type_labels'] = torch.stack(batch_type_labels)
            batch_dict['target_positions'] = torch.stack(batch_target_positions)
        
        return batch_dict
