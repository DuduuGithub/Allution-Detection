import torch
from torch.utils.data import Dataset
import re
from transformers import BertTokenizer
import csv
from torch.utils.data import DataLoader
from config import ALLUSION_TYPES_PATH



def load_allusion_types(file_path):
    """从CSV文件加载典故类型映射"""
    type_label2id = {}
    id2type_label = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)
        types = []
        for line in f:
            if not line.strip():
                continue
            allusion_name = line.strip().split('\t')[0] 
            types.append(allusion_name)
    
    # 创建双向映射
    for idx, type_label in enumerate(sorted(types)):
        type_label2id[type_label] = idx
        id2type_label[idx] = type_label
    
    return type_label2id, id2type_label
    
class PoetryNERDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len, task='position'):
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
        self.type_label2id, self.id2type_label = load_allusion_types(ALLUSION_TYPES_PATH)
        
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
        
        for positions, allusion_type in allusions:
            if positions:  
                position_labels[positions[0]] = 'B'
                for pos in positions[1:]:
                    position_labels[pos] = 'I'
                
                for pos in positions:
                    type_labels[pos] = allusion_type
        
        # 将标签转换为ID，如果没有对应标签，则使用-1 为id
        position_ids = [self.position_label2id[label] for label in position_labels]
        type_ids = [self.type_label2id.get(label, -1) for label in type_labels]
        
        return position_ids, type_ids
    
    def __getitem__(self, idx):
        """获取单个样本"""
        item = self.data[idx]
        text = item['text']
        position_labels = item['position_labels']
        type_labels = item['type_labels'] if 'type_labels' in item else None
        
        # tokenizer处理
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 确保标签长度与输入一致
        if len(position_labels) > self.max_len:
            position_labels = position_labels[:self.max_len]
        else:
            position_labels = position_labels + [0] * (self.max_len - len(position_labels))
        
        # 准备返回数据
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'position_labels': torch.tensor(position_labels, dtype=torch.long)
        }
        
        # 如果是类型分类任务且有类型标签
        if self.task == 'type' and type_labels is not None:
            if len(type_labels) > self.max_len:
                type_labels = type_labels[:self.max_len]
            else:
                type_labels = type_labels + [0] * (self.max_len - len(type_labels))
            result['type_labels'] = torch.tensor(type_labels, dtype=torch.long)
        
        return result
    
    def __len__(self):
        return len(self.data)
    
    def read_data(self, file_path):
        """读取数据文件"""
        dataset = []
        total_lines = 0
        
        print(f"开始读取数据文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过标题行
            header = next(f)    # next会读取迭代器所在行的内容，迭代器移动到下一行 ···
            print(f"文件头: {header.strip()}")
            
            for line_num, line in enumerate(f, 1): # enumerate会把一个可迭代对象，选择一个起始值，获得一个产生两个值：索引 元素 的迭代器···
                total_lines += 1
                if not line.strip() or '\t' not in line:
                    continue
                    
                # 解析行数据
                text, allusions = self.parse_line(line)
                
                # 创建标签序列
                position_labels, type_labels = self.create_labels(text, allusions)
                
                dataset.append({
                    'text': text,
                    'position_labels': position_labels,
                    'type_labels': type_labels
                })
        
        print(f"\n数据加载统计:")
        print(f"成功加载样本数: {len(dataset)}")
        print(f"\n数据加载完成···:")
        return dataset
