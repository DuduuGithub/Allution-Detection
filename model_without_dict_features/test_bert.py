import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加项目根目录到路径

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from TorchCRF import CRF
from without_dict_features.config import BERT_MODEL_PATH
import torch.nn.functional as F

class AllusionBERTCRF(nn.Module):
    
    #num_types: 类型数量 需要在使用时通过建立allution_types.txt的映射关系的同时获得
    def __init__(self, bert_path, num_types, bi_label_weight,position_weight):
        """
        Args:
            bert_path: BERT模型路径
            num_types: 典故类型数量
            dict_size: 典故词典大小
            bi_label_weight: B/I标签的权重（相对于O标签）,在计算损失时，B/I标签的权重为1+bi_label_weight，O标签的权重为1
        """
        super().__init__()
        # B/I标签的权重（用于位置识别任务）
        self.bi_label_weight = nn.Parameter(torch.tensor(bi_label_weight))
        
        # 联合训练的权重参数（用于联合损失计算）
        self.position_weight = position_weight
        
        # BERT基础模型
        self.bert = BertModel.from_pretrained(bert_path)

        self.dropout = nn.Dropout(0.1)
        
        # 位置识别模块 (B/I/O)
        self.position_classifier = nn.Linear(self.bert.config.hidden_size, 3)
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
        
        # 设置合法转移为更大的正值
        transitions[1, 2] = 50.0  # B -> I 更强烈地鼓励
        transitions[0, 1] = 20.0  # O -> B 更强烈地鼓励
        transitions[2, 2] = 30.0  # I -> I 更强烈地鼓励
        
        # 添加O->O的适度惩罚，避免模型过度预测O
        transitions[0, 0] = -5.0
        
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
            nn.Linear(self.bert.config.hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        '''
            注意力层的作用：
            如对于[塞翁失马]
                塞: 0.3
                翁: 0.2
                失: 0.4
                马: 0.1
            将权重与原始特征相乘并求和
        '''
        
        # 类别识别模块
        self.type_classifier = nn.Linear(self.bert.config.hidden_size, num_types)

    

    def attention_pooling(self, hidden_states,start,end):
        """
        注意力池化层
        Args:
            hidden_states: [span_length, hidden_size*2]
        Returns:
            pooled_features: [hidden_size*2]
        """
        
        
        span = hidden_states[start:end]  # 选取当前样本的跨度
        attention_weights = self.attention(span)  # [span_length, 1]
        attention_weights = torch.softmax(attention_weights, dim=0)  # 归一化注意力权重
        pooled_feature = torch.sum(span * attention_weights, dim=0)  # 加权求和        
        return pooled_feature

    def weighted_crf_loss(self, emissions, labels, mask):
        """
        计算带权重的CRF损失
        Args:
            emissions: [batch_size, seq_len, num_tags]
            labels: [batch_size, seq_len]
            mask: [batch_size, seq_len]
        """        
        # 创建权重矩阵 [batch_size, seq_len, 1]
        weights = torch.where(labels > 0, 
                             torch.tensor(1 + self.bi_label_weight, device=labels.device),
                             torch.ones(1, device=labels.device))
        
        # 直接将权重扩展到所有标签维度 [batch_size, seq_len, num_tags]
        weights = weights.unsqueeze(-1).expand_as(emissions)
        
        # 计算加权发射分数 [batch_size, seq_len, num_tags]
        weighted_emissions = emissions * weights
        
        # 计算CRF损失
        loss = -self.position_crf(
            weighted_emissions,
            labels,
            mask=mask,
            reduction='none'
        )
        
        return loss.mean()


    def type_classification_loss(self, logits, labels, label_smoothing=0.1, gamma=2.0):
        """
        改进的类型分类损失函数
        Args:
            logits: [batch_type_nums, num_types]
            labels: [batch_type_nums]
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

    def forward(self, input_ids, attention_mask,
                position_labels=None, type_labels=None, target_positions=None,train_mode=True,task='position'):
        """
        联合训练前向传播
        Args:
            input_ids: 输入的token ids
            attention_mask: 注意力掩码 已考虑[CLS]和[SEP]
            position_labels: 位置标签 (B/I/O)    已考虑[CLS]和[SEP]
            type_labels:[batch_size, max_type_len] 类型标签    已考虑[CLS]和[SEP]
            target_positions: 待判断词的位置索引 [batch_size, max_type_len,2]  已考虑[CLS]和[SEP]
        
        预测模式return:
        'position_labels': [[0, 1, 2, 0, ...]],  # 整句话的位置标签预测
        'type_predictions': [
            (start1, end1, [(type_id1, prob1), (type_id2, prob2), ...]),  # 第一个典故
            (start2, end2, [(type_id1, prob1), (type_id2, prob2), ...]),  # 第二个典故
            ...
        ]
        """
        # 1. BERT编码
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_len, 768]
        
        
        # 5. 位置识别任务
        position_emissions = self.position_classifier(sequence_output)
        position_preds = self.position_crf.decode(position_emissions, mask=attention_mask.bool())
        
        # 训练模式
        # 训练模式的返回
        if train_mode == True:
            # 使用 attention_mask 作为损失计算的掩码
            mask = attention_mask.bool()  # [batch_size, seq_len]
            
            # 计算位置识别损失
            position_loss = self.weighted_crf_loss(position_emissions, position_labels, mask)
            
            # 计算类型识别损失
            batch_size = type_labels.size(0)
            max_type_len = type_labels.size(1)
            
            batch_type_preds = []
            batch_type_labels = []
            for batch_idx in range(batch_size):
                for type_idx in range(max_type_len):
                    if target_positions[batch_idx][type_idx].sum() > 0:  # 跳过填充的位置
                        # print('target_positions in forward:',target_positions[batch_idx][type_idx])
                        start = target_positions[batch_idx][type_idx][0].item()
                        end = target_positions[batch_idx][type_idx][1].item()
                        
                        pooled_features = self.attention_pooling(               #shape:[hidden_size * 2]
                            sequence_output[batch_idx],start,end
                        )
                        
                        type_logits = self.type_classifier(pooled_features)     #shape:[num_types]
                        batch_type_preds.append(type_logits)
                        batch_type_labels.append(type_labels[batch_idx][type_idx])
            # 将列表转换为张量
            batch_type_preds = torch.stack(batch_type_preds)    #shape:[batch_type_nums,num_types]
            batch_type_labels = torch.tensor(batch_type_labels, device=batch_type_preds.device)  # [num_samples]
            
            # print('batch_type_preds:',batch_type_preds)
            # print('batch_type_labels:',batch_type_labels)
            type_loss = self.type_classification_loss(
                batch_type_preds,
                batch_type_labels
            )
            
            type_loss*=10 # 为了保持类别识别和位置识别的数量级相同
            
            # 使用动态权重计算联合损失
            joint_loss = (self.position_weight * position_loss + 
                          (1 - self.position_weight) * type_loss)
            return {
                'loss': joint_loss,
                'position_loss': position_loss.item(),
                'type_loss': type_loss.item(),
                'joint_loss': joint_loss,
                'position_weight': self.position_weight,
                'type_weight': (1 - self.position_weight)
            }
        elif train_mode == False:
            # 预测模式
            # 1. 位置识别任务
            if task == 'position':  
                position_emissions = self.position_classifier(sequence_output)
                position_preds = self.position_crf.decode(position_emissions, mask=attention_mask.bool())

                # 去除CLS和SEP标记，并只保留实际文本长度
                position_preds_cleaned = []
                for pred, mask in zip(position_preds, attention_mask):
                    # 计算实际序列长度（减去CLS和SEP）
                    seq_len = mask.sum().item() - 2  # -2 for CLS and SEP
                    # 只保留实际文本部分（去除CLS、SEP和padding）
                    position_preds_cleaned.append(pred[1:seq_len+1])
                return {
                    'position_predictions': position_preds_cleaned
                }
            elif task == 'type':    
                # 2. 类型识别任务
                type_predictions = []
                
                batch_size = target_positions.size(0)
                max_type_len = target_positions.size(1)
                
                # 获取原始文本
                tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
                for batch_idx in range(batch_size):
                    for type_idx in range(max_type_len):
                        if target_positions[batch_idx][type_idx].sum() > 0:  # 跳过填充的位置
                            start = target_positions[batch_idx][type_idx][0].item()
                            end = target_positions[batch_idx][type_idx][1].item()
                            
                            pooled_features = self.attention_pooling(               #shape:[hidden_size * 2]
                                sequence_output[batch_idx],start,end
                            )
                            
                            type_logits = self.type_classifier(pooled_features)     #shape:[num_types]
                            
                            # 计算概率
                            type_probs = F.softmax(type_logits, dim=-1)
                            
                            # 获取top5
                            top5_probs, top5_indices = torch.topk(type_probs, k=min(5, type_probs.size(-1)))
                            
                            
                            # 将预测结果和概率组合
                            all_position_types = []
                            for indice, prob in zip(top5_indices, top5_probs):
                                all_position_types.append((indice.item(), prob.item()))  # 使用 .item() 提取值
                                
                            type_predictions.append((start-1, end-1, all_position_types))  # -1 因为[CLS]


                return {
                    'type_predictions': type_predictions
                }



