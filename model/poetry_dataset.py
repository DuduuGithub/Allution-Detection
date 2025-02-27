import torch
from torch.utils.data import Dataset
import re
from transformers import BertTokenizer
import csv
from torch.utils.data import DataLoader
import json

def load_sentence_mappings(mapping_path):
    """
    加载句子映射
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    return mappings['sentence_to_id'], mappings['id_to_sentence']

class PoetryNERDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len, type_label2id, id2type_label, 
                 task='position', features_path=None, mapping_path=None):
        """
        Args:
            file_path: 数据文件路径
            tokenizer: BERT tokenizer
            max_len: 最大序列长度
            task: 'position' 用于典故位置识别, 'type' 用于典故类型分类
            features_path: 预处理特征文件路径
            mapping_path: 句子映射文件路径
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.type_label2id = type_label2id
        self.id2type_label = id2type_label
        self.task = task
        
        # 典故位置标注标签映射
        self.position_label2id = {
            'O': 0,
            'B': 1,
            'I': 2
        }
        
        # 加载句子映射和预处理的特征
        if mapping_path and features_path:
            self.sentence_to_id, _ = load_sentence_mappings(mapping_path)
            self.precomputed_features = torch.load(features_path)
        else:
            self.sentence_to_id = None
            self.precomputed_features = None
            
        # 使用原有的read_data方法加载和处理数据
        self.data = self.read_data(file_path)
        
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
                
                text, position_labels, type_labels = result
                
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
        
        # 获取预处理的特征
        if self.precomputed_features is not None:
            sent_id = self.sentence_to_id[text]
            dict_features = self.precomputed_features[sent_id]
            # 转换回原始数据类型
            dict_features = {
                'indices': dict_features['indices'].to(torch.long),
                'values': dict_features['values'].float(),
                'active_counts': dict_features['active_counts'].to(torch.long)
            }
        else:
            dict_features = None
        
        # BERT tokenization
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'text': text,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        
        # 添加字典特征,有个待解决的问题是，当该诗句不存在时，dict_features为None，且未处理
        if dict_features:
            result['dict_features'] = dict_features
        else:
            result['dict_features'] = None
        
        # 添加任务相关的标签
        if self.task == 'position':
            position_labels = item['position_labels']
            padded_position_labels = torch.zeros(len(result['input_ids']), dtype=torch.long)
            padded_position_labels[1:len(text)+1] = torch.tensor(position_labels)
            result['position_labels'] = padded_position_labels
        else:
            type_labels = item['type_labels']
            target_positions = item['target_positions']
            padded_type_labels = torch.zeros(len(result['input_ids']), dtype=torch.long)
            padded_type_labels[1:len(text)+1] = torch.tensor(type_labels)
            result['type_labels'] = padded_type_labels
            result['target_positions'] = torch.tensor([pos + 1 for pos in target_positions])
            
        return result

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        """自定义批处理函数，处理不同长度的特征"""
        # 获取这个batch中的最大文本长度
        max_text_len = max(len(item['text']) for item in batch)
        max_seq_len = max_text_len + 2  # 加2是为了CLS和SEP
        print(f"max_seq_len: {max_seq_len}")
        
        
        # 准备batch数据
        batch_texts = []
        batch_input_ids = []
        batch_attention_mask = []
        batch_position_labels = []
        
        # 准备字典特征的列表
        indices_list = []
        values_list = []
        active_counts_list = []
        
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
            
            # 处理字典特征
            if item['dict_features'] is not None:
                indices = item['dict_features']['indices'][:max_seq_len]
                values = item['dict_features']['values'][:max_seq_len]
                active_counts = item['dict_features']['active_counts'][:max_seq_len]
                
                print(f"text: {text}")
                print(f"indices: {indices}")
                print(f"values: {values}")
                print(f"active_counts: {active_counts}")
            else:
                # 如果没有特征，创建空特征
                # 注意：第一个位置([CLS])设为0
                indices = torch.zeros((max_seq_len, 5), dtype=torch.long)
                values = torch.zeros((max_seq_len, 5), dtype=torch.float)
                active_counts = torch.zeros(max_seq_len, dtype=torch.long)
                
                # 只为实际文本部分创建特征（跳过[CLS]）
                text_len = len(text)
                indices[1:text_len+1] = torch.zeros((text_len, 5), dtype=torch.long)
                values[1:text_len+1] = torch.zeros((text_len, 5), dtype=torch.float)
                active_counts[1:text_len+1] = torch.zeros(text_len, dtype=torch.long)

                
                
            # 补全到最大长度（保持[CLS]位置为0）
            if indices.size(0) < max_seq_len:
                pad_len = max_seq_len - indices.size(0)
                indices = torch.cat([indices, torch.zeros((pad_len, 5), dtype=torch.long)], dim=0)
                values = torch.cat([values, torch.zeros((pad_len, 5), dtype=torch.float)], dim=0)
                active_counts = torch.cat([active_counts, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            
            indices_list.append(indices)
            values_list.append(values)
            active_counts_list.append(active_counts)
            
            if self.task == 'position':
                position_labels = item['position_labels']
                padded_labels = torch.zeros(max_seq_len, dtype=torch.long)
                padded_labels[1:len(text)+1] = position_labels[1:len(text)+1]  # 跳过[CLS]
                batch_position_labels.append(padded_labels)
        
        # 将列表转换为张量
        batch_dict = {
            'text': batch_texts,
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
            'dict_features': {
                'indices': torch.stack(indices_list),
                'values': torch.stack(values_list),
                'active_counts': torch.stack(active_counts_list)
            }
        }
        
        if self.task == 'position':
            batch_dict['position_labels'] = torch.stack(batch_position_labels)
        
        return batch_dict
