import torch
from torch.utils.data import Dataset
import re
from transformers import BertTokenizer
import csv

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
        
        # 首先收集所有典故类型并创建映射
        self.collect_allusion_types()
        
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
    import torch
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer
    import os
    
    def test_dataset_loading():
        print("开始测试数据集加载...")
        
        # 初始化tokenizer
        tokenizer = BertTokenizer.from_pretrained('model/guwenbert-large')
        
        # 创建训练集
        train_dataset = PoetryNERDataset(
            file_path='data/final_data.csv',
            tokenizer=tokenizer,
            max_len=128
        )
        
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True
        )
        
        print(f"\n数据集基本信息:")
        print(f"数据集大小: {len(train_dataset)}")
        print(f"位置标签映射: {train_dataset.position_label2id}")
        print(f"类型标签数量: {len(train_dataset.type_label2id)}")
        
        # 测试遍历整个数据集
        print("\n开始遍历数据集...")
        total_batches = len(train_loader)
        
        try:
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx % 100 == 0:  # 每100个batch打印一次进度
                    print(f"处理进度: {batch_idx}/{total_batches} batches")
                
                # 检查batch中的数据
                assert 'input_ids' in batch
                assert 'attention_mask' in batch
                assert 'position_labels' in batch
                assert 'type_labels' in batch
                
                # 检查数据维度
                assert batch['input_ids'].shape[0] == batch['attention_mask'].shape[0]
                assert batch['input_ids'].shape[0] == batch['position_labels'].shape[0]
                assert batch['input_ids'].shape[0] == batch['type_labels'].shape[0]
            
            print("\n数据集测试完成！所有数据都可以正确加载。")
            
            # 展示第一个batch的详细信息
            first_batch = next(iter(train_loader))
            print("\n第一个batch的详细信息:")
            for key, value in first_batch.items():
                print(f"\n{key}:")
                print(f"Shape: {value.shape}")
                print(f"Type: {value.dtype}")
                if batch_idx == 0:  # 只打印第一个batch的具体内容
                    print(f"Content: {value}")
            
        except Exception as e:
            print(f"\n数据集测试失败！错误信息：")
            print(str(e))
            raise e

    # 运行测试
    test_dataset_loading()