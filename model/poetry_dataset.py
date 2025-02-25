import torch
from torch.utils.data import Dataset
import re
from transformers import BertTokenizer
import csv
from torch.utils.data import DataLoader
from config import ALLUSION_TYPES_PATH
    
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
        """解析单行数据，提取诗句和典故位置"""
        
        parts = line.strip().split('\t')  
        
        #诗句提取      
        sentence = parts[0]
                
        #典故提取
        allusion_info = parts[6].strip()  
        allusion_parts = allusion_info.split(';')
        allusions = []
        for part in allusion_parts:
            part = part.strip()
            if not part:  # 跳过空字符串
                continue

            part = part.strip('[]')
            items = [item.strip() for item in part.split(',')]        
            positions = [int(pos) for pos in items[:-1]]
            allusion_type = items[-1]
            
            allusions.append((positions, allusion_type))
            
        return sentence, allusions
        
    
    def create_labels(self, text, allusions):
        """根据典故位置创建BIO标签和类型标签"""
        position_labels = ['O'] * len(text) 
        type_labels = ['O'] * len(text)      
        
        # 如果没有典故（variation_number为0），直接返回全O标签
        if not allusions:
            return [self.position_label2id['O']] * len(text), [-1] * len(text)
        
        for positions, allusion_type in allusions:
            if positions:  
                position_labels[positions[0]] = 'B'
                for pos in positions[1:]:
                    position_labels[pos] = 'I'
                
                for pos in positions:
                    type_labels[pos] = allusion_type
        
        # 将标签转换为ID
        position_ids = [self.position_label2id[label] for label in position_labels]
        type_ids = [self.type_label2id.get(label, -1) for label in type_labels]
        
        return position_ids, type_ids
    
    def read_data(self, file_path):
        """读取数据文件"""
        dataset = []
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # 跳过标题行
            for line in f:
                if not line.strip() or '\t' not in line:
                    continue
                text, allusions = self.parse_line(line)
                position_labels, type_labels = self.create_labels(text, allusions)
                
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
        item = self.data[idx]
        text = item['text']
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': text  # 添加原始文本
        }
        
        if self.task == 'position':
            position_labels = item['position_labels']
            if len(position_labels) > self.max_len:
                position_labels = position_labels[:self.max_len]
            else:
                position_labels = position_labels + [0] * (self.max_len - len(position_labels))
            
            result['position_labels'] = torch.tensor(position_labels, dtype=torch.long)
            
        else: 
            result['target_positions'] = torch.tensor(item['target_positions'], dtype=torch.long)
            result['type_labels'] = torch.tensor(item['target_type'], dtype=torch.long)
            
        return result

    def __len__(self):
        return len(self.data)
