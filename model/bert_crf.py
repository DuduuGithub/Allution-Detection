import torch
import torch.nn as nn
from transformers import BertModel
from TorchCRF import CRF

from config import BERT_MODEL_PATH
class AllusionBERTCRF(nn.Module):
    def __init__(self, num_types, task='position'):
        super(AllusionBERTCRF, self).__init__()
        
        self.bert = BertModel.from_pretrained(BERT_MODEL_PATH)
        self.dropout = nn.Dropout(0.1)
        
        # 修改位置识别分类器为3分类：B/I/O
        self.position_classifier = nn.Linear(self.bert.config.hidden_size, 3)
        self.position_crf = CRF(3)  # 修改为3个标签状态
        
        # 类型分类器
        self.type_classifier = nn.Linear(self.bert.config.hidden_size, num_types)
        self.type_crf = CRF(num_types)
        
        self.task = task
    
    def forward(self, input_ids, attention_mask, labels=None, type_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        # 位置识别 (B/I/O)
        position_emissions = self.position_classifier(sequence_output)
        mask = attention_mask.bool()
        
        if self.task == 'position':
            if labels is not None:
                loss = -self.position_crf(position_emissions, labels, mask=mask)
                return loss.mean()
            else:
                prediction = self.position_crf.viterbi_decode(position_emissions, mask=mask)
                return prediction
        
        # 类型分类
        elif self.task == 'type':
            # 获取CRF预测结果
            position_pred = self.position_crf.viterbi_decode(position_emissions, mask=mask)
            
            # 根据attention_mask获取每个序列的实际长度
            seq_lengths = attention_mask.sum(dim=1).tolist()
            
            # 将预测结果填充到与输入相同的长度
            batch_size, max_len = input_ids.shape
            padded_position_pred = torch.zeros((batch_size, max_len), 
                                            dtype=torch.long, 
                                            device=input_ids.device)
            
            # 对每个序列单独处理
            for i, (pred, length) in enumerate(zip(position_pred, seq_lengths)):
                # 确保预测结果不超过实际序列长度
                pred_length = min(len(pred), length)
                # 只填充有效长度的部分
                padded_position_pred[i, :pred_length] = torch.tensor(
                    pred[:pred_length], 
                    dtype=torch.long, 
                    device=input_ids.device
                )
            
            # 类型分类
            type_emissions = self.type_classifier(sequence_output)
            
            if type_labels is not None:
                # 只在预测为典故的位置(B或I)计算类型损失
                position_loss = -self.position_crf(position_emissions, labels, mask=mask)
                
                # 创建类型预测的mask（只在典故位置B或I计算损失）
                type_mask = ((labels == 1) | (labels == 2)) & mask  # B=1, I=2
                type_loss = -self.type_crf(type_emissions, type_labels, mask=type_mask)
                
                # 计算平均损失
                total_loss = position_loss.mean() + type_loss.mean()
                return total_loss
            else:
                # 使用 viterbi_decode 替换 decode
                type_pred = self.type_crf.viterbi_decode(type_emissions, mask=mask)
                return padded_position_pred, type_pred
