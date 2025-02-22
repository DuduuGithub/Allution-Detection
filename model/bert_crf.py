import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from TorchCRF import CRF
import difflib

from config import BERT_MODEL_PATH

class AllusionBERTCRF(nn.Module):
    
    #num_types: 类型数量 需要在使用时通过建立allution_types.txt的映射关系的同时获得
    def __init__(self, bert_path, num_types,task):
        super(AllusionBERTCRF, self).__init__()
        self.task = task
        # BERT基础模型
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # BiLSTM层
        self.lstm_hidden_size = 256
        self.num_lstm_layers = 2
        self.bilstm = nn.LSTM(
            input_size=self.bert_hidden_size + self.dict_size,  # BERT输出 + 字典特征向量
            hidden_size=self.lstm_hidden_size,
            num_layers=self.num_lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # 位置识别模块 (B/I/O)
        self.position_classifier = nn.Linear(self.lstm_hidden_size * 2, 3)
        self.position_crf = CRF(3, batch_first=True)
        
        # 类别识别模块
        self.type_classifier = nn.Linear(self.lstm_hidden_size * 2, num_types)

    def forward(self, input_ids, attention_mask, dict_features, target_positions=None, position_labels=None, type_labels=None):
        """
        前向传播
        Args:
            input_ids: 输入的token ids
            attention_mask: 注意力掩码
            dict_features: 预处理得到的字典特征向量 [batch_size, seq_len, dict_size]
            target_positions: 待判断词的位置索引 [batch_size, 2] (start, end)
            position_labels: 位置标签 (B/I/O)
            type_labels: 类型标签
        """
        # BERT编码
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_len, bert_hidden]
        
        # 特征拼接
        combined_features = torch.cat([sequence_output, dict_features], dim=-1)
        
        # BiLSTM处理
        lstm_output, _ = self.bilstm(combined_features)
        
        if self.task == 'position':
            # 位置识别 (B/I/O)
            position_emissions = self.position_classifier(lstm_output)
            mask = attention_mask.bool()
            
            if position_labels is not None:
                loss = -self.position_crf(position_emissions, position_labels, mask=mask)
                return loss.mean()
            else:
                prediction = self.position_crf.viterbi_decode(position_emissions, mask=mask)
                return prediction
        
        elif self.task == 'type':
            # 获取目标位置的特征
            batch_size = input_ids.size(0)
            target_features = []
            
            # 对每个样本提取目标位置的特征
            for i in range(batch_size):
                start, end = target_positions[i]
                # 提取目标位置的特征并进行平均池化
                span_features = lstm_output[i, start:end+1].mean(dim=0)
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
