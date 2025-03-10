import torch
from torch.utils.data import Dataset
import re
from transformers import BertTokenizer
import csv
from torch.utils.data import DataLoader
import json
import random  # 添加在文件开头的import部分

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
                
                if self.task == 'position':
                    text, position_labels = result
                    dataset.append({
                        'text': text,
                        'position_labels': position_labels,
                    })
                else:  # task == 'type'
                    text, target_positions, type_label = result
                    dataset.append({
                        'text': text,
                        'target_positions': target_positions,
                        'target_type': type_label
                    })
        
        return dataset

    def parse_line(self, line):
        """解析单行数据，提取诗句和标签"""
        parts = line.strip().split('\t')
        try:
            # 获取基本信息
            text = parts[0].strip()
            
            if self.task == 'position':
                # 位置识别任务
                variation_number = int(parts[4])
                
                # 如果没有典故，直接返回全O标签
                if variation_number == 0:
                    position_labels = [self.position_label2id['O']] * len(text)
                    return text, position_labels
                
                # 处理有典故的情况
                allusion_info = parts[6].strip()  # transformed_allusion列
                allusion_parts = allusion_info.split(';')
                
                # 初始化标签序列
                position_labels = ['O'] * len(text)
                
                # 处理每个典故
                for part in allusion_parts:
                    part = part.strip()
                    if not part:
                        continue
                    
                    part = part.strip('[]')
                    items = [item.strip() for item in part.split(',')]
                    positions = [int(pos) for pos in items[:-1]]
                    
                    # 设置位置标签
                    if positions:
                        position_labels[positions[0]] = 'B'
                        for pos in positions[1:]:
                            position_labels[pos] = 'I'
                
                # 将标签转换为ID
                position_ids = [self.position_label2id[label] for label in position_labels]
                return text, position_ids
                
            else:  # task == 'type'
                # 类型识别任务
                variation_number = int(parts[4])
                
                # 如果是不包含典故的样本
                if variation_number == 0:
                    # 随机选择一个长度在2到4之间的片段
                    text_length = len(text)
                    if text_length < 2:
                        return None
                    
                    # 确定随机片段的长度
                    span_length = random.randint(2, min(3, text_length))
                    # 确定随机片段的起始位置
                    max_start = text_length - span_length
                    if max_start < 0:
                        return None
                    start_pos = random.randint(0, max_start)
                    
                    # 返回文本、随机选择的起始和结束位置、类型标签为0（表示非典故）
                    return text, (start_pos, start_pos + span_length - 1), 0
                
                allusion = parts[3].strip()  # allusion列
                allusion_index = parts[5].strip()  # allusion_index列
                
                if not allusion_index:  # 如果没有典故位置
                    return None
                
                # 解析典故位置
                try:
                    positions = eval(allusion_index)  # 例如 "[[4, 5]]"
                    if not positions or not isinstance(positions, list):
                        return None
                    
                    # 获取第一组位置（因为数据集中每行只有一个位置组）
                    pos_group = positions[0]  # 获取第一个（也是唯一的）位置组
                    if not pos_group:  # 如果位置组为空
                        return None
                    
                    # 设置类型标签
                    type_label = self.type_label2id.get(allusion, -1)
                    if type_label == -1:  # 如果找不到对应的类型
                        return None
                    
                    # 返回文本、起始和结束位置（使用同一个位置组）
                    return text, (pos_group[0], pos_group[-1]), type_label
                    
                except (ValueError, SyntaxError):
                    return None
                
        except (IndexError, ValueError):
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
        
        # BERT tokenization 在此处加上cls和sep
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
        else:  # task == 'type'
            # 只需要目标位置和类型标签
            target_positions = item['target_positions']
            result['target_positions'] = torch.tensor([pos + 1 for pos in target_positions])  # +1 因为CLS
            result['type_labels'] = torch.tensor([item['target_type']])  # 包装成一维张量
        
        return result

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        """自定义批处理函数，处理不同长度的特征"""
        # 获取这个batch中的最大文本长度
        max_text_len = max(len(item['text']) for item in batch)
        max_seq_len = max_text_len + 2  # 加2是为了CLS和SEP

        
        
        # 准备batch数据
        batch_texts = []
        batch_input_ids = []
        batch_attention_mask = []
        batch_position_labels = []
        
        batch_type_labels = []
        batch_target_positions = []
        indices_list = []
        values_list = []
        active_counts_list = []
        
        # 获取这个batch中的最大序列长度
        max_seq_len = max(len(item['input_ids']) for item in batch)
        
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
            
            # 处理字典特征
            if item['dict_features'] is not None:
                indices = item['dict_features']['indices'][:max_seq_len]
                values = item['dict_features']['values'][:max_seq_len]
                active_counts = item['dict_features']['active_counts'][:max_seq_len]
                
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
            else:  # task == 'type'
                batch_type_labels.append(item['type_labels'])
                batch_target_positions.append(item['target_positions'])
        
        # 构建返回字典
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
        else:  # task == 'type'
            batch_dict['type_labels'] = torch.stack(batch_type_labels)
            batch_dict['target_positions'] = torch.stack(batch_target_positions)
        
        return batch_dict
