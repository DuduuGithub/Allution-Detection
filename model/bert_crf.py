import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from TorchCRF import CRF
from scipy.sparse import csr_matrix
from config import BERT_MODEL_PATH
from difflib import SequenceMatcher
from config import OPTIMAL_EPS,min_samples_size

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
            
        # indices张量示例 (max_active=5)
        indices[0] = [
            [1, 0, 0, 0, 0],  # 位置0：只有1个典故
            [2, 3, 4, 0, 0],  # 位置1：有3个典故
            [0, 0, 0, 0, 0],  # 位置2：没有典故
            [5, 6, 7, 8, 0],  # 位置3：有4个典故
            [9, 0, 0, 0, 0]   # 位置4：有1个典故
        ]

        # values张量示例
        values[0] = [
            [0.9, 0.0, 0.0, 0.0, 0.0],  # 位置0
            [0.8, 0.7, 0.6, 0.0, 0.0],  # 位置1
            [0.0, 0.0, 0.0, 0.0, 0.0],  # 位置2
            [0.9, 0.8, 0.7, 0.6, 0.0],  # 位置3
            [0.9, 0.0, 0.0, 0.0, 0.0]   # 位置4
        ]

        # active_counts记录实际数量
        active_counts[0] = [1, 3, 0, 4, 1]
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
        
        # 4. 对每个位置的特征求和 在多个典故的维度上求和
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
            position_labels: 位置标签 (B/I/O)    已考虑[CLS]和[SEP]
            type_labels: 类型标签    已考虑[CLS]和[SEP]
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
                # 忽略CLS和SEP的损失计算
                loss = -self.position_crf(
                    position_emissions[:, 1:-1, :],  # 去掉CLS和SEP
                    position_labels[:, 1:-1],        # 去掉CLS和SEP
                    mask=mask[:, 1:-1]               # 去掉CLS和SEP
                )
                return loss.mean()
            else:
                # 预测时也忽略CLS和SEP
                prediction = self.position_crf.viterbi_decode(
                    position_emissions[:, 1:-1, :],
                    mask=mask[:, 1:-1]
                )
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
    
    # 初始化每个位置的典故匹配列表
    position_to_allusions = [[] for _ in range(seq_len)]  # 每个位置可能的典故列表
    
    # 为每个典故分配ID
    allusion_to_id = {name: idx for idx, name in enumerate(allusion_dict.keys())}
    
    def get_variants(variants):
        """获取变体列表"""
        if isinstance(variants, list):
            # 如果是列表但只有一个元素且是字符串，可能需要进一步解析
            if len(variants) == 1 and isinstance(variants[0], str):
                try:
                    # 尝试解析可能的嵌套列表
                    parsed = eval(variants[0])
                    if isinstance(parsed, list):
                        return parsed
                except:
                    pass
            return variants
            
        if isinstance(variants, str):
            try:
                # 尝试解析字符串形式的列表
                parsed = eval(variants)
                if isinstance(parsed, list):
                    # 如果解析出的是列表，递归处理可能的嵌套
                    return get_variants(parsed)
                return [variants]
            except:
                # 如果解析失败，按分号分割
                return variants.split(';')
                
        return []  # 兜底返回空列表
    
    # 处理每个样本
    for b, text in enumerate(batch_texts):
        # 对每个起始位置
        for start_pos in range(len(text)-1):
            max_len = min(5, len(text) - start_pos)
            
            # 尝试不同长度的窗口
            for window_size in range(2, max_len + 1):
                context = text[start_pos:start_pos + window_size]
                end_pos = start_pos + window_size - 1
                
                # 匹配典故
                for allusion_name, variants in allusion_dict.items():
                    variant_list = get_variants(variants)
                    max_similarity = 0
                    
                    # 找出该典故在当前位置的最高相似度
                    for variant in variant_list:
                        similarity = SequenceMatcher(None, context, variant).ratio()
                        if similarity > max_similarity:
                            max_similarity = similarity
                            
                    
                    # 计算距离
                    distance = 1 - max_similarity
                    if distance < OPTIMAL_EPS:
                        # 将该典故添加到范围内的每个位置
                        for pos in range(start_pos, end_pos + 1):
                            position_to_allusions[pos].append((
                                allusion_to_id[allusion_name],
                                max_similarity
                            ))
        
        # 为每个位置选择最佳的max_active个典故
        indices = torch.zeros((batch_size, seq_len, max_active), dtype=torch.long)
        values = torch.zeros((batch_size, seq_len, max_active), dtype=torch.float)
        active_counts = torch.zeros((batch_size, seq_len), dtype=torch.long)
        
        for pos in range(seq_len):
            # 对该位置的所有典故按相似度排序
            matches = position_to_allusions[pos]
            matches.sort(key=lambda x: x[1], reverse=True)
            matches = matches[:max_active]  # 只保留最佳的max_active个
            
            # 记录匹配数量和结果
            active_counts[b, pos] = len(matches)
            for idx, (allusion_id, similarity) in enumerate(matches):
                indices[b, pos, idx] = allusion_id
                values[b, pos, idx] = similarity
    
    # 在返回前添加CLS和SEP的处理
    padded_indices = torch.zeros((batch_size, seq_len + 2, max_active), dtype=torch.long)
    padded_values = torch.zeros((batch_size, seq_len + 2, max_active), dtype=torch.float)
    padded_active_counts = torch.zeros((batch_size, seq_len + 2), dtype=torch.long)
    
    # 将原始特征放在中间位置（跳过CLS）
    padded_indices[:, 1:seq_len+1, :] = indices
    padded_values[:, 1:seq_len+1, :] = values
    padded_active_counts[:, 1:seq_len+1] = active_counts
    
    return {
        'indices': padded_indices,
        'values': padded_values,
        'active_counts': padded_active_counts
    }

