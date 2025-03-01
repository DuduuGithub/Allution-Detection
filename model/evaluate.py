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
    
    # 分别计算O标签、B标签和I标签的准确率
    o_mask = (torch.tensor(all_labels) == 0) & (torch.tensor(all_labels) != -1)
    b_mask = (torch.tensor(all_labels) == 1) & (torch.tensor(all_labels) != -1)
    i_mask = (torch.tensor(all_labels) == 2) & (torch.tensor(all_labels) != -1)

    predictions_tensor = torch.tensor(all_predictions, device=torch.tensor(all_labels).device)

    # O标签指标
    o_correct = ((predictions_tensor == torch.tensor(all_labels)) & o_mask).sum().item()
    o_total = o_mask.sum().item()
    o_pred_total = (predictions_tensor == 0).sum().item()  # 预测为O的总数

    # B标签指标
    b_correct = ((predictions_tensor == torch.tensor(all_labels)) & b_mask).sum().item()
    b_total = b_mask.sum().item()
    b_pred_total = (predictions_tensor == 1).sum().item()  # 预测为B的总数

    # I标签指标
    i_correct = ((predictions_tensor == torch.tensor(all_labels)) & i_mask).sum().item()
    i_total = i_mask.sum().item()
    i_pred_total = (predictions_tensor == 2).sum().item()  # 预测为I的总数

    # 计算精确率和召回率
    o_precision = o_correct / o_pred_total if o_pred_total > 0 else 0
    o_recall = o_correct / o_total if o_total > 0 else 0
    b_precision = b_correct / b_pred_total if b_pred_total > 0 else 0
    b_recall = b_correct / b_total if b_total > 0 else 0
    i_precision = i_correct / i_pred_total if i_pred_total > 0 else 0
    i_recall = i_correct / i_total if i_total > 0 else 0

    # 打印详细信息
    print('\nDetailed Metrics:')
    print(f'O-tag - Precision: {o_precision*100:.2f}%, Recall: {o_recall*100:.2f}%, Accuracy: {o_correct/o_total*100:.2f}% ({o_correct}/{o_total})')
    print(f'B-tag - Precision: {b_precision*100:.2f}%, Recall: {b_recall*100:.2f}%, Accuracy: {b_correct/b_total*100:.2f}% ({b_correct}/{b_total})')
    print(f'I-tag - Precision: {i_precision*100:.2f}%, Recall: {i_recall*100:.2f}%, Accuracy: {i_correct/i_total*100:.2f}% ({i_correct}/{i_total})')
    print(f'Overall Accuracy: {(o_correct + b_correct + i_correct)/(o_total + b_total + i_total)*100:.2f}%')
    
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