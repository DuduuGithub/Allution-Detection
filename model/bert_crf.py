import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加项目根目录到路径

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from TorchCRF import CRF
from scipy.sparse import csr_matrix
from model.config import BERT_MODEL_PATH, OPTIMAL_EPS, min_samples_size
from difflib import SequenceMatcher
import torch.nn.functional as F

class AllusionBERTCRF(nn.Module):
    
    #num_types: 类型数量 需要在使用时通过建立allution_types.txt的映射关系的同时获得
    def __init__(self, bert_path, num_types, dict_size, pos_weight=150):
        super(AllusionBERTCRF, self).__init__()
        self.pos_weight = pos_weight  # 添加权重参数
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
        
        # 添加CRF转移约束
        transitions = torch.zeros(3, 3)
        
        # 设置不合法转移为极小值
        # O -> I 不允许
        transitions[0, 2] = -10000.0
        # I -> B 不建议（可以有但不推荐）
        transitions[2, 1] = -50.0
        # B -> O 不建议（可以有但不推荐）
        transitions[1, 0] = -50.0
        
        # 设置合法转移为正值以鼓励
        # B -> I 鼓励
        transitions[1, 2] = 20.0
        # O -> B 鼓励
        transitions[0, 1] = 8.0
        # I -> I 鼓励
        transitions[2, 2] = 15.0
        
        # 开始标签到各个标签的转移
        # START -> O 允许
        self.position_crf.start_transitions.data[0] = 0
        # START -> B 允许
        self.position_crf.start_transitions.data[1] = 0
        # START -> I 不允许
        self.position_crf.start_transitions.data[2] = -10000.0
        
        # 各个标签到结束标签的转移
        # O -> END 允许
        self.position_crf.end_transitions.data[0] = 0
        # B -> END 不建议
        self.position_crf.end_transitions.data[1] = -100.0
        # I -> END 允许
        self.position_crf.end_transitions.data[2] = 0
        
        # 设置转移矩阵
        self.position_crf.transitions.data = transitions
        
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

    def weighted_crf_loss(self, emissions, labels, mask):
        """
        计算带权重的CRF损失
        Args:
            emissions: [batch_size, seq_len, num_tags]
            labels: [batch_size, seq_len]
            mask: [batch_size, seq_len]
        """
        batch_size, seq_len = labels.shape
        
        # 创建权重矩阵
        weights = torch.ones_like(labels, dtype=torch.float)
        # 将典故位置(B/I)的权重设置为pos_weight
        weights = torch.where(labels > 0, torch.tensor(self.pos_weight, device=labels.device), weights)
        
        # 计算基础CRF损失
        base_loss = -self.position_crf(
            emissions,
            labels,
            mask=mask,
            reduction='none'
        )
        
        # 打印调试信息（确保只看mask内的有效标签）
        # masked_labels = labels[mask]
        # masked_weights = weights[mask]
        # print("Labels distribution in loss:", torch.bincount(masked_labels))
        # print("Original weights values:", torch.unique(masked_weights))
        # print("Number of O labels:", (masked_labels == 0).sum().item())
        # print("Number of B labels:", (masked_labels == 1).sum().item())
        # print("Number of I labels:", (masked_labels == 2).sum().item())
        # print("Number of weighted positions:", (masked_weights == pos_weight).sum().item())
        
        weighted_loss = base_loss * weights.mean(dim=1)
        return weighted_loss.mean()

    def batch_attention_pooling(self, hidden_states, start_positions, end_positions):
        """
        批量注意力池化层
        Args:
            hidden_states: [batch_size, seq_len, hidden_size*2]
            start_positions: [batch_size]
            end_positions: [batch_size]
        Returns:
            pooled_features: [batch_size, hidden_size*2]
        """
        batch_size = hidden_states.size(0)
        max_span_length = (end_positions - start_positions + 1).max()
        
        # 创建批处理掩码
        span_masks = torch.arange(max_span_length, device=hidden_states.device)[None, :] < \
                    (end_positions - start_positions + 1)[:, None]  # [batch_size, max_span_length]
        
        # 收集所有跨度的特征
        batch_spans = []
        for i in range(batch_size):
            span = hidden_states[i, start_positions[i]:end_positions[i]+1]
            # 填充到最大长度
            if span.size(0) < max_span_length:
                padding = torch.zeros(max_span_length - span.size(0), span.size(1), 
                                    device=span.device)
                span = torch.cat([span, padding], dim=0)
            batch_spans.append(span)
        
        batch_spans = torch.stack(batch_spans)  # [batch_size, max_span_length, hidden_size*2]
        
        # 计算注意力分数
        attention_weights = self.attention(batch_spans)  # [batch_size, max_span_length, 1]
        
        # 应用掩码
        attention_weights = attention_weights.masked_fill(~span_masks.unsqueeze(-1), float('-inf'))
        attention_weights = torch.softmax(attention_weights, dim=1)  # [batch_size, max_span_length, 1]
        
        # 加权求和
        weighted_sum = torch.sum(batch_spans * attention_weights, dim=1)  # [batch_size, hidden_size*2]
        
        return weighted_sum

    def type_classification_loss(self, logits, labels, label_smoothing=0.1, gamma=2.0):
        """
        改进的类型分类损失函数
        Args:
            logits: [batch_size, num_types]
            labels: [batch_size]
            label_smoothing: 标签平滑参数
            gamma: 焦点损失参数
        """
        # 标签平滑
        num_classes = logits.size(-1)
        smooth_labels = torch.zeros_like(logits).scatter_(
            1, labels.unsqueeze(1), 1-label_smoothing
        ) + label_smoothing/num_classes
        
        # 计算焦点损失
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        ce_loss = -(smooth_labels * log_probs).sum(dim=-1)  # 交叉熵
        pt = torch.sum(smooth_labels * probs, dim=-1)
        focal_loss = ce_loss * (1 - pt).pow(gamma)
        
        return focal_loss.mean()

    def forward(self, input_ids, attention_mask, dict_features, task='position', 
                position_labels=None, type_labels=None, target_positions=None):
        """
        前向传播
        Args:
            input_ids: 输入的token ids
            attention_mask: 注意力掩码 已考虑[CLS]和[SEP]
            dict_features: 稀疏特征字典
            task: 'position' 或 'type'，指定当前任务
            position_labels: 位置标签 (B/I/O)    已考虑[CLS]和[SEP]
            type_labels: 类型标签    已考虑[CLS]和[SEP]
            target_positions: 待判断词的位置索引 [batch_size, 2] (start, end) 已考虑[CLS]和[SEP]
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
            
            if position_labels is not None:
                # 训练模式
                batch_size, seq_len = position_labels.shape  # 使用position_labels的形状
                mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=position_emissions.device)
                
                # 根据attention_mask设置实际长度
                for i in range(batch_size):
                    valid_len = attention_mask[i].sum().item()
                    mask[i, valid_len:] = False
                
                
                # 使用带权重的损失
                loss = self.weighted_crf_loss(
                    position_emissions,
                    position_labels,
                    mask=mask
                )
                return loss
            
            else:
                # 预测模式 - 添加dummy样本
                batch_size, seq_len = position_emissions.shape[:2]
                dummy_emission = position_emissions[0:1].clone()
                position_emissions = torch.cat([dummy_emission, position_emissions], dim=0)
                
                # 创建包含dummy样本的mask
                mask = torch.ones((batch_size + 1, seq_len), dtype=torch.bool, device=position_emissions.device)
                mask[0] = True  # dummy样本的mask全为True
                
                # print('attention_mask:', attention_mask)
                
                # 为实际样本设置mask
                for i in range(batch_size):
                    valid_len = attention_mask[i].sum().item()
                    # print('valid_len:', valid_len)
                    mask[i+1, valid_len:] = False
                
                # print('mask:', mask)

                # 预测
                prediction = self.position_crf.decode(
                    position_emissions,
                    mask=mask
                )
                
                # 去掉dummy样本并确保所有序列长度一致
                predictions = []
                for pred in prediction[1:]:  # 跳过dummy样本
                    # 如果预测长度小于seq_len，补充到完整长度
                    if len(pred) < seq_len:
                        pred = pred + [0] * (seq_len - len(pred))
                    predictions.append(pred[1:-1])  # 去掉[CLS]和[SEP]
                
                # print('Prediction in forward:', predictions)
                return predictions
        
        elif task == 'type':
            # 使用批处理版本的注意力池化
            batch_size = input_ids.size(0)
            start_positions = target_positions[:, 0]
            end_positions = target_positions[:, 1]
            target_features = self.batch_attention_pooling(
                lstm_output, start_positions, end_positions
            )
            
            # 类型分类
            type_logits = self.type_classifier(target_features)  # [batch_size, num_types]
            
            if type_labels is not None:
                # 训练模式：使用改进的损失函数
                # 将 type_labels 从 [batch_size, 1] 转换为 [batch_size]
                type_labels = type_labels.squeeze(1)
                loss = self.type_classification_loss(type_logits, type_labels)
                return loss
            else:
                # 预测模式：返回 top-5 预测结果和概率
                probs = F.softmax(type_logits, dim=-1)
                top5_probs, top5_indices = torch.topk(probs, k=5, dim=-1)
                return {
                    'predictions': top5_indices,
                    'probabilities': top5_probs
                }

    def freeze_shared_parameters(self):
        """冻结共享参数（BERT和BiLSTM层）"""
        # 冻结BERT
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # 冻结BiLSTM
        for param in self.bilstm.parameters():
            param.requires_grad = False
        
        # 冻结字典特征相关层
        for param in self.dict_embedding.parameters():
            param.requires_grad = False
        for param in self.dict_transform.parameters():
            param.requires_grad = False

    def verify_frozen_parameters(self):
        """验证共享参数是否被正确冻结"""
        for name, param in self.named_parameters():
            if any(layer in name for layer in ['bert', 'bilstm', 'dict_embedding', 'dict_transform']):
                if param.requires_grad:
                    print(f"Warning: {name} should be frozen but is not!")
            elif any(layer in name for layer in ['type_classifier', 'attention']):
                if not param.requires_grad:
                    print(f"Warning: {name} should be trainable but is frozen!")

    def adjust_shared_learning_rate(self, epoch, base_lr):
        """根据训练轮数调整共享参数的学习率"""
        # 前10个epoch保持较大学习率以快速学习
        if epoch < 10:
            return base_lr
        # 之后逐渐降低学习率
        else:
            return base_lr * 0.1

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

