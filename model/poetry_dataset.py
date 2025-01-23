import torch
from torch.utils.data import Dataset
import re
from transformers import BertTokenizer

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
        self.type_label2id = self.load_type_mapping('data/allusion_types.txt')
        
        # 典故位置标注标签映射
        self.position_label2id = {
            'O': 0,
            'B': 1,
            'I': 2
        }
        
        # 最后加载数据
        self.data = self.read_data(file_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task = task
    
    def load_type_mapping(self, type_file):
        """加载典故类型映射"""
        type_label2id = {'O': 0}  # 非典故位置的标签
        
        # 收集所有典故类型
        allusion_types = set()
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ']' in line:
                    text, annotations = line.split('[', 1)
                    pattern = r'\((\d+),(\d+),([^)]+)\)'
                    matches = re.finditer(pattern, annotations)
                    for match in matches:
                        allusion_type = match.group(3)
                        allusion_types.add(allusion_type)
        
        # 创建类型到ID的映射
        for idx, allusion_type in enumerate(sorted(allusion_types), 1):
            type_label2id[allusion_type] = idx
        
        # 保存映射到文件
        with open(type_file, 'w', encoding='utf-8') as f:
            for allusion_type in sorted(allusion_types):
                f.write(f"{allusion_type}\n")
        
        return type_label2id
    
    def parse_line(self, line):
        """解析单行数据，提取诗句、典故位置和类型"""
        # 分离诗句和标注信息
        text, annotations = line.split('[', 1)
        text = text.strip()
        
        # 提取所有典故标注
        allusions = []
        pattern = r'\((\d+),(\d+),([^)]+)\)'
        matches = re.finditer(pattern, annotations)
        for match in matches:
            start, end, allusion_type = match.groups()
            allusions.append((int(start), int(end), allusion_type))
        
        return text, allusions
    
    def create_labels(self, text, allusions):
        """根据典故位置创建BIO标签和类型标签"""
        position_labels = ['O'] * len(text)
        type_labels = ['O'] * len(text)
        
        for start, end, allusion_type in allusions:
            # 设置BIO标签
            position_labels[start] = 'B'
            for i in range(start + 1, end):
                position_labels[i] = 'I'
            
            # 设置类型标签 (所有典故位置都使用相同的类型ID)
            type_id = self.type_label2id.get(allusion_type, 0)  # 如果找不到类型，使用0
            for i in range(start, end):
                type_labels[i] = allusion_type
        
        return position_labels, type_labels
    
    def read_data(self, file_path):
        """读取数据文件"""
        dataset = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip() or ']' not in line:
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
        
        return dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        position_labels = item['position_labels']
        type_labels = item['type_labels']
        
        # 对诗句进行编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 转换标签为id
        position_ids = [self.position_label2id[label] for label in position_labels]
        type_ids = [self.type_label2id[label] for label in type_labels]
        
        # 填充标签序列
        if len(position_ids) < self.max_len:
            position_ids = position_ids + [0] * (self.max_len - len(position_ids))
            type_ids = type_ids + [0] * (self.max_len - len(type_ids))
        else:
            position_ids = position_ids[:self.max_len]
            type_ids = type_ids[:self.max_len]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'position_labels': torch.tensor(position_ids),
            'type_labels': torch.tensor(type_ids)
        } 

if __name__ == "__main__":
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('model/guwenbert-large')
    
    # 创建数据集实例
    dataset = PoetryNERDataset(
        file_path='data/example.txt',  # 你的数据文件路径
        tokenizer=tokenizer,
        max_len=128
    )
    
    # 测试数据集基本信息
    print(f"数据集大小: {len(dataset)}")
    print(f"\n典故位置标签映射: {dataset.position_label2id}")
    print(f"\n典故类型标签映射: {dataset.type_label2id}")
    
    # 测试单个样本的处理
    sample = dataset[0]
    print("\n第一个样本的处理结果:")
    for key, value in sample.items():
        print(f"\n{key}:")
        print(f"Shape: {value.shape}")
        print(f"Content: {value}")