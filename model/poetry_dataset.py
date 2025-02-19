import torch
from torch.utils.data import Dataset
import re
from transformers import BertTokenizer
import csv
from torch.utils.data import DataLoader
from utils import load_allusion_types
from config import ALLUSION_TYPES_PATH

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
        
        # 然后加载数据
        self.data = self.read_data(file_path)
    
    def collect_allusion_types(self):
        """首先遍历数据文件收集所有典故类型"""
        print("开始收集典故类型...")
        allusion_types = set(['O'])  # 包含默认的'O'标签
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # 跳过标题行
            next(f)
            for line in f:
                if '\t' not in line:
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 7 or parts[4].strip() == '0':
                    continue
                
                try:
                    # 解析位置信息字符串
                    allusion_info = parts[6].strip()
                    if not allusion_info or allusion_info == '[]':
                        continue
                    
                    # 按分号分割多个典故
                    allusion_parts = allusion_info.split(';')
                    
                    for part in allusion_parts:
                        part = part.strip()
                        if not part:
                            continue
                        
                        # 移除首尾的方括号
                        part = part.strip('[]')
                        # 分割位置和典故名称
                        items = [item.strip() for item in part.split(',')]
                        
                        if len(items) < 3:
                            continue
                        
                        # 获取典故名称
                        allusion_type = items[-1].strip('"\'')
                        allusion_types.add(allusion_type)
                        
                except Exception as e:
                    continue
        
        # 创建类型到ID的映射
        self.type_label2id = {label: idx for idx, label in enumerate(sorted(allusion_types))}
        
        print(f"共收集到 {len(allusion_types)-1} 种典故类型")
        
        # 保存映射到文件（可选）
        with open('data/allusion_types.txt', 'w', encoding='utf-8') as f:
            for allusion_type in sorted(allusion_types):
                if allusion_type != 'O':
                    f.write(f"{allusion_type}\n")
    
    def parse_line(self, line):
        """解析单行数据，提取诗句和典故位置"""
        parts = line.strip().split('\t')        
        sentence = parts[0]
        variation_number = parts[4].strip()  # 获取典故数量
        
        # 如果典故数量为0，跳过该行
        if variation_number == '0':
            return None, []
        
        try:
            # 解析位置信息字符串
            allusion_info = parts[6].strip()  # transformed_allusion
            # 按分号分割多个典故
            allusion_parts = allusion_info.split(';')
            
            allusions = []
            for part in allusion_parts:
                part = part.strip()
                if not part:  # 跳过空字符串
                    continue
                    
                # 移除首尾的方括号
                part = part.strip('[]')
                # 分割位置和典故名称
                items = [item.strip() for item in part.split(',')]        
                # 获取位置和典故名称
                positions = [int(pos) for pos in items[:-1]]  # 所有数字都是位置
                allusion_type = items[-1] # 最后一个是典故名称
                
                allusions.append((positions, allusion_type))
                
            return sentence, allusions
        
        except Exception as e:
            print(f"解析典故信息失败: {parts[6]}")
            print(f"错误信息: {str(e)}")
            return None, []
    
    def create_labels(self, text, allusions):
        """根据典故位置创建BIO标签和类型标签"""
        position_labels = ['O'] * len(text)
        type_labels = ['O'] * len(text)
        
        for positions, allusion_type in allusions:
            # positions 包含了典故的所有位置
            if positions:  # 确保有位置信息
                # 第一个位置标记为B
                position_labels[positions[0]] = 'B'
                # 其余位置标记为I
                for pos in positions[1:]:
                    position_labels[pos] = 'I'
                
                # 所有位置都标记为相应的典故类型
                for pos in positions:
                    type_labels[pos] = allusion_type
        
        return position_labels, type_labels
    
    def read_data(self, file_path):
        """读取数据文件"""
        dataset = []
        error_count = 0
        total_lines = 0
        
        print(f"开始读取数据文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过标题行
            header = next(f)
            print(f"文件头: {header.strip()}")
            
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                if not line.strip() or '\t' not in line:
                    continue
                    
                # 解析行数据
                text, allusions = self.parse_line(line)
                if text is None:  # 跳过格式不正确的行
                    error_count += 1
                    if error_count < 5:  # 只打印前几个错误的行
                        print(f"行 {line_num} 解析失败: {line.strip()}")
                    continue
                
                # 创建标签序列
                position_labels, type_labels = self.create_labels(text, allusions)
                
                dataset.append({
                    'text': text,
                    'position_labels': position_labels,
                    'type_labels': type_labels
                })
                
                if len(dataset) < 3:  # 打印前几个成功解析的样本
                    print(f"\n成功解析的样本 {len(dataset)}:")
                    print(f"文本: {text}")
                    print(f"典故: {allusions}")
                    print(f"位置标签: {position_labels}")
                    print(f"类型标签: {type_labels}")
        
        print(f"\n数据加载统计:")
        print(f"总行数: {total_lines}")
        print(f"成功加载样本数: {len(dataset)}")
        print(f"解析失败行数: {error_count}")
        
        if len(dataset) == 0:
            raise ValueError("没有成功加载任何数据！请检查数据文件格式是否正确。")
        
        return dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        position_labels = item['position_labels']
        
        # tokenizer处理
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 准备返回数据
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'position_labels': torch.tensor(position_labels)
        }
        
        # 如果是类型分类任务，添加类型标签
        if self.task == 'type' and 'type_labels' in item:
            # 确保使用统一的类型映射
            type_labels = [self.type_label2id.get(t, 0) for t in item['type_labels']]
            result['type_labels'] = torch.tensor(type_labels)
            
        return result

def test_dataset():
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('model/guwenbert-large')
    
    # 创建数据集
    dataset = PoetryNERDataset(
        file_path='data/merged_data.csv',
        tokenizer=tokenizer,
        max_len=128,
        task='type'  # 使用type任务进行测试
    )
    
    # 创建DataLoader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 获取一个batch并检查
    print("\n获取第一个batch进行检查...")
    batch = next(iter(dataloader))
    
    # 检查batch中的数据
    print("\nBatch 数据形状:")
    for key, value in batch.items():
        print(f"{key}: {value.shape}")
    
    # 检查position_labels的值
    print("\nPosition labels 的值分布:")
    position_labels = batch['position_labels']
    unique_values, counts = torch.unique(position_labels, return_counts=True)
    for value, count in zip(unique_values.tolist(), counts.tolist()):
        print(f"标签 {value} 出现次数: {count}")
    
    # 检查每个序列的实际长度
    print("\n检查每个序列的实际长度:")
    attention_mask = batch['attention_mask']
    sequence_lengths = attention_mask.sum(dim=1)
    print(f"序列长度统计: {sequence_lengths.tolist()}")
    
    # 检查是否存在长度不一致的问题
    print("\n检查数据一致性:")
    print(f"input_ids 形状: {batch['input_ids'].shape}")
    print(f"attention_mask 形状: {batch['attention_mask'].shape}")
    print(f"position_labels 形状: {batch['position_labels'].shape}")
    
    # 打印几个完整的序列示例
    print("\n打印前3个序列的详细信息:")
    for i in range(min(3, batch_size)):
        print(f"\n序列 {i+1}:")
        print(f"实际长度: {sequence_lengths[i]}")
        print(f"Position labels: {batch['position_labels'][i][:sequence_lengths[i]].tolist()}")
        # 解码文本
        tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
        text = tokenizer.convert_tokens_to_string(tokens)
        print(f"文本: {text}")

if __name__ == "__main__":
    test_dataset()