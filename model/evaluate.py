import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from poetry_dataset import PoetryNERDataset, load_allusion_types
from bert_crf import AllusionBERTCRF
from config import BERT_MODEL_PATH, TEST_PATH, MAX_SEQ_LEN, SAVE_DIR, ALLUSION_TYPES_PATH
import os
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np


def evaluate_model(model, dataloader, device, task='position'):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_labels = batch['position_labels'].to(device)
            
            # 获取实际序列长度
            seq_lengths = attention_mask.sum(dim=1).tolist()
            
            if task == 'position':
                predictions = model(input_ids, attention_mask)
                # predictions 是一个列表，需要转换为张量并填充
                batch_size, max_len = input_ids.shape
                padded_predictions = torch.zeros((batch_size, max_len), 
                                              dtype=torch.long, 
                                              device=device)
                for i, (pred, length) in enumerate(zip(predictions, seq_lengths)):
                    pred_length = min(len(pred), length)
                    padded_predictions[i, :pred_length] = torch.tensor(
                        pred[:pred_length], 
                        dtype=torch.long, 
                        device=device
                    )
                predictions = padded_predictions
            else:  # task == 'type'
                position_pred, type_pred = model(input_ids, attention_mask)
                # type_pred 是一个列表，需要转换为张量
                batch_size, max_len = input_ids.shape
                padded_predictions = torch.zeros((batch_size, max_len), 
                                              dtype=torch.long, 
                                              device=device)
                for i, (pred, length) in enumerate(zip(type_pred, seq_lengths)):
                    pred_length = min(len(pred), length)
                    padded_predictions[i, :pred_length] = torch.tensor(
                        pred[:pred_length], 
                        dtype=torch.long, 
                        device=device
                    )
                predictions = padded_predictions
                position_labels = batch['type_labels'].to(device)  # 使用类型标签
            
            # 只考虑非填充部分的预测和标签
            for pred, label, length in zip(predictions, position_labels, seq_lengths):
                # 现在 pred 已经是张量了，可以使用 cpu()
                all_predictions.extend(pred[:length].cpu().tolist())
                all_labels.extend(label[:length].cpu().tolist())
    
    # 计算指标
    # 对于位置识别任务，标签为 0(O), 1(B), 2(I)
    # 对于类型分类任务，标签为实际的类型ID
    labels = list(range(3)) if task == 'position' else None
    report = classification_report(
        all_labels, 
        all_predictions, 
        labels=labels,
        target_names=['O', 'B', 'I'] if task == 'position' else None,
        digits=4
    )
    
    # 计算整体指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_predictions, 
        average='weighted'
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'detailed_report': report
    }

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    # 创建测试数据集
    test_dataset = PoetryNERDataset(TEST_PATH, tokenizer, MAX_SEQ_LEN, task='type')
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 获取典故类型数量
    type_label2id, _ = load_allusion_types(ALLUSION_TYPES_PATH)
    num_types = len(type_label2id)
    print(f"Total number of allusion types: {num_types}")
    
    # 评估位置识别模型
    print("\nEvaluating Position Detection Model...")
    position_model = AllusionBERTCRF(num_types=num_types, task='position').to(device)
    position_checkpoint = torch.load(os.path.join(SAVE_DIR, 'best_model_position.pt'))
    position_model.load_state_dict(position_checkpoint['model_state_dict'])
    position_metrics = evaluate_model(position_model, test_dataloader, device, 'position')
    
    # 评估类型分类模型
    print("\nEvaluating Type Classification Model...")
    type_model = AllusionBERTCRF(num_types=num_types, task='type').to(device)
    type_checkpoint = torch.load(os.path.join(SAVE_DIR, 'best_model_type.pt'))
    type_model.load_state_dict(type_checkpoint['model_state_dict'])
    type_metrics = evaluate_model(type_model, test_dataloader, device, 'type')
    
    # 打印结果
    print("\nPosition Detection Results:")
    print(f"Precision: {position_metrics['precision']:.4f}")
    print(f"Recall: {position_metrics['recall']:.4f}")
    print(f"F1 Score: {position_metrics['f1']:.4f}")
    print("\nDetailed Position Detection Report:")
    print(position_metrics['detailed_report'])
    
    print("\nType Classification Results:")
    print(f"Precision: {type_metrics['precision']:.4f}")
    print(f"Recall: {type_metrics['recall']:.4f}")
    print(f"F1 Score: {type_metrics['f1']:.4f}")
    print("\nDetailed Type Classification Report:")
    print(type_metrics['detailed_report'])

if __name__ == '__main__':
    main() 