import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from TorchCRF import CRF
from scipy.sparse import csr_matrix
from config import BERT_MODEL_PATH, OPTIMAL_EPS
from difflib import SequenceMatcher

class AllusionBERTCRF(nn.Module):
    
    #num_types: 类型数量 需要在使用时通过建立allution_types.txt的映射关系的同时获得
    def __init__(self, bert_path, num_types, dict_size):
        super(AllusionBERTCRF, self).__init__()
        # BERT基础模型
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.dict_size = dict_size
        
        # 字典特征嵌入层
        self.dict_embedding = nn.Embedding(dict_size, 256)
        self.dict_transform = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # BiLSTM层
        self.lstm_hidden_size = 256
        self.num_lstm_layers = 2
        self.bilstm = nn.LSTM(
            input_size=self.bert_hidden_size + 256,  # BERT输出 + 字典特征
            hidden_size=self.lstm_hidden_size,
            num_layers=self.num_lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # 位置识别模块 (B/I/O)
        self.position_classifier = nn.Linear(self.lstm_hidden_size * 2, 3)
        self.position_crf = CRF(3, batch_first=True)
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 2, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # 类别识别模块
        self.type_classifier = nn.Linear(self.lstm_hidden_size * 2, num_types)

    def process_dict_features(self, dict_features):
        """处理稀疏的字典特征
        
        Args:
            dict_features: 字典 {
                'indices': [batch_size, seq_len, max_active],  # 非零特征的索引
                'values': [batch_size, seq_len, max_active],   # 对应的值
                'active_counts': [batch_size, seq_len],        # 每个位置的非零特征数量
            }
        Returns:
            tensor: [batch_size, seq_len, 256]
        """
        batch_size, seq_len, max_active = dict_features['indices'].shape
        
        # 1. 展平处理
        flat_indices = dict_features['indices'].view(-1, max_active)  # [batch_size*seq_len, max_active]
        flat_values = dict_features['values'].view(-1, max_active)    # [batch_size*seq_len, max_active]
        flat_counts = dict_features['active_counts'].view(-1)         # [batch_size*seq_len]
        
        # 2. 获取嵌入
        # [batch_size*seq_len, max_active, 256]
        embedded_features = self.dict_embedding(flat_indices)
        
        # 3. 应用权重
        # [batch_size*seq_len, max_active, 256]
        weighted_features = embedded_features * flat_values.unsqueeze(-1)
        
        # 4. 对每个位置的特征求和
        # [batch_size*seq_len, 256]
        summed_features = weighted_features.sum(dim=1)
        
        # 5. 重塑回原始维度
        # [batch_size, seq_len, 256]
        reshaped_features = summed_features.view(batch_size, seq_len, -1)
        
        # 6. 特征转换
        transformed_features = self.dict_transform(reshaped_features)
        
        return transformed_features

    def attention_pooling(self, hidden_states):
        """
        注意力池化层
        Args:
            hidden_states: [span_length, hidden_size*2]
        Returns:
            pooled_features: [hidden_size*2]
        """
        # 计算注意力分数
        attention_weights = self.attention(hidden_states)  # [span_length, 1]
        attention_weights = torch.softmax(attention_weights, dim=0)  # [span_length, 1]
        
        # 加权求和
        weighted_sum = torch.sum(hidden_states * attention_weights, dim=0)  # [hidden_size*2]
        
        return weighted_sum

    def forward(self, input_ids, attention_mask, dict_features, task='position', 
                position_labels=None, type_labels=None, target_positions=None):
        """
        前向传播
        Args:
            input_ids: 输入的token ids
            attention_mask: 注意力掩码
            dict_features: 稀疏特征字典
            task: 'position' 或 'type'，指定当前任务
            position_labels: 位置标签 (B/I/O)
            type_labels: 类型标签
            target_positions: 待判断词的位置索引 [batch_size, 2] (start, end)
        """
        # 1. BERT编码
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_len, 768]
        
        # 2. 处理稀疏字典特征
        dict_output = self.process_dict_features(dict_features)  # [batch_size, seq_len, 256]
        
        # 3. 特征拼接
        combined_features = torch.cat([sequence_output, dict_output], dim=-1)
        
        # 4. BiLSTM处理
        lstm_output, _ = self.bilstm(combined_features)
        
        if task == 'position':
            # 位置识别 (B/I/O)
            position_emissions = self.position_classifier(lstm_output)
            mask = attention_mask.bool()
            
            if position_labels is not None:
                loss = -self.position_crf(position_emissions, position_labels, mask=mask)
                return loss.mean()
            else:
                prediction = self.position_crf.viterbi_decode(position_emissions, mask=mask)
                return prediction
        
        elif task == 'type':
            # 获取目标位置的特征
            batch_size = input_ids.size(0)
            target_features = []
            
            # 对每个样本提取目标位置的特征
            for i in range(batch_size):
                start, end = target_positions[i]
                # 使用注意力池化
                span_features = self.attention_pooling(lstm_output[i, start:end+1])
                target_features.append(span_features)
            
            target_features = torch.stack(target_features)  # [batch_size, lstm_hidden*2]
        
            # 类型分类
            type_logits = self.type_classifier(target_features)  # [batch_size, num_types]
            
            if type_labels is not None:
                loss = nn.CrossEntropyLoss()(type_logits, type_labels)
                return loss
            else:
                type_pred = torch.argmax(type_logits, dim=-1)
                return type_pred

def prepare_sparse_features(batch_texts, allusion_dict, max_active=5):
    """将文本批量转换为稀疏特征格式"""
    batch_size = len(batch_texts)
    seq_len = max(len(text) for text in batch_texts)
    
    # 初始化存储张量
    indices = torch.zeros((batch_size, seq_len, max_active), dtype=torch.long)
    values = torch.zeros((batch_size, seq_len, max_active), dtype=torch.float)
    active_counts = torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    # 为每个典故分配ID
    allusion_to_id = {name: idx for idx, name in enumerate(allusion_dict.keys())}
    
    # 处理每个样本
    for b, text in enumerate(batch_texts):
        for pos in range(len(text)):
            context = text[max(0, pos-2):min(len(text), pos+3)]
            
            # 匹配典故
            matches = []
            for allusion_name, variants in allusion_dict.items():
                max_similarity = max(
                    SequenceMatcher(None, context, variant).ratio()
                    for variant in variants
                )
                
                # 使用config.py中定义的OPTIMAL_EPS
                if max_similarity > OPTIMAL_EPS:
                    matches.append((
                        allusion_to_id[allusion_name],
                        max_similarity
                    ))
            
            # 按相似度排序并取top-k
            matches.sort(key=lambda x: x[1], reverse=True)
            matches = matches[:max_active]
            
            # 记录匹配数量和结果
            active_counts[b, pos] = len(matches)
            for idx, (allusion_id, similarity) in enumerate(matches):
                indices[b, pos, idx] = allusion_id
                values[b, pos, idx] = similarity
    
    return {
        'indices': indices,
        'values': values,
        'active_counts': active_counts
    }

