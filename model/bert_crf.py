import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

from config import BERT_MODEL_PATH
class AllusisonBERTCRF(nn.Module):
    def __init__(self, num_types, task='position'):
        super(AllusisonBERTCRF, self).__init__()
        
        self.bert = BertModel.from_pretrained(BERT_MODEL_PATH)
        self.dropout = nn.Dropout(0.1)
        
        # 修改位置识别分类器为3分类：B/I/O
        self.position_classifier = nn.Linear(self.bert.config.hidden_size, 3)
        self.position_crf = CRF(3, batch_first=True)  # 修改为3个标签状态
        
        # 类型分类器
        self.type_classifier = nn.Linear(self.bert.config.hidden_size, num_types)
        self.type_crf = CRF(num_types, batch_first=True)
        
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
                loss = -self.position_crf(position_emissions, labels, mask=mask, reduction='mean')
                return loss
            else:
                prediction = self.position_crf.decode(position_emissions, mask=mask)
                return prediction
        
        # 类型分类
        elif self.task == 'type':
            # 获取位置预测
            position_pred = self.position_crf.decode(position_emissions, mask=mask)
            position_pred = torch.tensor(position_pred, device=input_ids.device)
            
            # 类型分类
            type_emissions = self.type_classifier(sequence_output)
            
            if type_labels is not None:
                # 只在预测为典故的位置(B或I)计算类型损失
                position_loss = -self.position_crf(position_emissions, labels, mask=mask, reduction='mean')
                
                # 创建类型预测的mask（只在典故位置B或I计算损失）
                type_mask = ((labels == 1) | (labels == 2)) & mask  # B=1, I=2
                type_loss = -self.type_crf(type_emissions, type_labels, mask=type_mask, reduction='mean')
                
                # 总损失为位置损失和类型损失的加权和
                total_loss = position_loss + type_loss
                return total_loss
            else:
                # 推理时，只对预测为典故的位置(B或I)进行类型预测
                type_pred = self.type_crf.decode(type_emissions, mask=mask)
                return position_pred, type_pred
