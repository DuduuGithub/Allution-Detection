import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from poetry_dataset import PoetryNERDataset, load_allusion_types, load_allusion_dict, prepare_sparse_features
from bert_crf import AllusionBERTCRF
from config import BERT_MODEL_PATH, TEST_PATH, MAX_SEQ_LEN, SAVE_DIR
import os
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
import argparse


def evaluate_model(model, dataloader, device, task='position'):
    model.eval()
    all_predictions = []
    all_labels = []
    
    # 加载典故词典
    allusion_dict, _, _, _ = load_allusion_dict()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 计算实际序列长度
            seq_lengths = attention_mask.sum(dim=1).tolist()
            
            # 添加字典特征
            texts = batch['text']
            dict_features = prepare_sparse_features(texts, allusion_dict)
            dict_features = {k: v.to(device) for k, v in dict_features.items()}
            
            if task == 'position':
                position_labels = batch['position_labels'].to(device)
                predictions = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    dict_features=dict_features,
                    task='position'
                )
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
                labels = position_labels
            else:  # task == 'type'
                type_labels = batch['type_labels'].to(device)
                target_positions = batch['target_positions'].to(device)
                predictions = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    dict_features=dict_features,
                    task='type',
                    target_positions=target_positions
                )
                labels = type_labels
            
            # 只考虑非填充部分的预测和标签
            for pred, label, length in zip(predictions, labels, seq_lengths):
                all_predictions.extend(pred[:length].cpu().tolist())
                all_labels.extend(label[:length].cpu().tolist())
    
    # 计算指标
    # 对于位置识别任务，标签为 0(O), 1(B), 2(I)
    # 对于类型分类任务，标签为实际的类型ID
    if task == 'position':
        labels = list(range(3))
        target_names = ['O', 'B', 'I']
    else:
        labels = None
        target_names = None
        
    report = classification_report(
        all_labels, 
        all_predictions, 
        labels=labels,
        target_names=target_names,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, choices=['position', 'type'], 
                       required=True, help='Evaluation stage: position or type')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载tokenizer和典故词典
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    allusion_dict, type_label2id, id2type_label, num_types = load_allusion_dict()
    
    # 创建测试数据集
    test_file = f'4_test_{args.stage}.csv'
    test_dataset = PoetryNERDataset(
        test_file, 
        tokenizer, 
        MAX_SEQ_LEN,
        type_label2id=type_label2id,
        id2type_label=id2type_label,
        task=args.stage
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Total number of allusion types: {num_types}")
    
    # 加载和评估模型
    model = AllusionBERTCRF(
        bert_model_path=BERT_MODEL_PATH,
        num_types=num_types,
        dict_size=len(allusion_dict)
    ).to(device)
    
    checkpoint = torch.load(os.path.join(SAVE_DIR, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nEvaluating {args.stage.capitalize()} Model...")
    metrics = evaluate_model(model, test_dataloader, device, args.stage)
    
    # 打印结果
    print(f"\n{args.stage.capitalize()} Results:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nDetailed Report:")
    print(metrics['detailed_report'])

if __name__ == '__main__':
    main() 