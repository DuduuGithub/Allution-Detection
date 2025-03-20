import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import BertTokenizer
from model.poetry_dataset import PoetryNERDataset
from model.bert_crf import AllusionBERTCRF
from model.train import load_allusion_dict
from model.config import (
    BERT_MODEL_PATH, MAX_SEQ_LEN, SAVE_DIR, 
    BATCH_SIZE, DATA_DIR
)
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

def load_models():
    """加载预训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    # 加载典故词典以获取类型数量
    allusion_dict, _, _, num_types = load_allusion_dict()
    dict_size = len(allusion_dict)
    
    # 创建两个模型实例
    position_model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size=dict_size).to(device)
    type_model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size=dict_size).to(device)
    
    # 分别加载两个模型的参数
    position_checkpoint = torch.load(f'{SAVE_DIR}/best_model_position.pt', map_location=device)
    position_model.load_state_dict(position_checkpoint['model_state_dict'])
    position_model.eval()
    
    type_checkpoint = torch.load(f'{SAVE_DIR}/best_model_type.pt', map_location=device)
    type_model.load_state_dict(type_checkpoint['model_state_dict'])
    type_model.eval()
    
    return position_model, type_model, tokenizer, device

def evaluate_position_task(model, dataloader, device):
    """评估位置识别任务"""
    all_predictions = []
    all_labels = []
    
    # 位置识别任务的统计变量 - 移到循环外面
    b_tp, b_fp, b_fn = 0, 0, 0
    i_tp, i_fp, i_fn = 0, 0, 0
    o_tp, o_fp, o_fn = 0, 0, 0
    
    pbar = tqdm(dataloader, desc="Evaluating position task")
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_labels = batch['position_labels'].to(device)  # 移到GPU
            
            dict_features = {
                'indices': batch['dict_features']['indices'].to(device),
                'values': batch['dict_features']['values'].to(device),
                'active_counts': batch['dict_features']['active_counts'].to(device)
            }
        
            
            # 获取损失
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_features=dict_features,
                task='position',
                position_labels=position_labels,
            )
            # 获取预测结果

            predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_features=dict_features,
                task='position'
            )
            
            
            # 计算指标（忽略[CLS], [SEP]和填充标记）
            mask = attention_mask[:, 1:-1].bool()
            masked_labels = torch.where(mask, position_labels[:, 1:-1], 
                                     torch.zeros_like(position_labels[:, 1:-1]))
            

            # 统计各类指标
            for pred, label in zip(predictions, masked_labels):
                for p, l in zip(pred, label.cpu().numpy()):  # 确保label转到CPU
                    # B标签统计
                    if l == 1:  # 真实标签是B
                        if p == 1: b_tp += 1
                        else: b_fn += 1
                    if p == 1:  # 预测标签是B
                        if l != 1: b_fp += 1
                    
                    # I标签统计
                    if l == 2:  # 真实标签是I
                        if p == 2: i_tp += 1
                        else: i_fn += 1
                    if p == 2:  # 预测标签是I
                        if l != 2: i_fp += 1
                    
                    # O标签统计
                    if l == 0:  # 真实标签是O
                        if p == 0: o_tp += 1
                        else: o_fn += 1
                    if p == 0:  # 预测标签是O
                        if l != 0: o_fp += 1
                
                all_predictions.extend(pred)
                all_labels.extend(label.cpu().numpy())
            
            # 更新进度条信息
            b_f1 = 2*b_tp/(2*b_tp+b_fp+b_fn) if (2*b_tp+b_fp+b_fn) > 0 else 0
            i_f1 = 2*i_tp/(2*i_tp+i_fp+i_fn) if (2*i_tp+i_fp+i_fn) > 0 else 0
            o_f1 = 2*o_tp/(2*o_tp+o_fp+o_fn) if (2*o_tp+o_fp+o_fn) > 0 else 0
            pbar.set_postfix({
                'B_f1': f"{b_f1:.4f}",
                'I_f1': f"{i_f1:.4f}",
                'O_f1': f"{o_f1:.4f}"
            })
    
    # 计算并打印每个标签的详细指标
    def calculate_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1
    
    print("\nPosition Task Detailed Metrics:")
    # B标签指标
    b_precision, b_recall, b_f1 = calculate_metrics(b_tp, b_fp, b_fn)
    print(f"B Label - Precision: {b_precision:.4f}, Recall: {b_recall:.4f}, F1: {b_f1:.4f}")
    print(f"         TP: {b_tp}, FP: {b_fp}, FN: {b_fn}")
    
    # I标签指标
    i_precision, i_recall, i_f1 = calculate_metrics(i_tp, i_fp, i_fn)
    print(f"I Label - Precision: {i_precision:.4f}, Recall: {i_recall:.4f}, F1: {i_f1:.4f}")
    print(f"         TP: {i_tp}, FP: {i_fp}, FN: {i_fn}")
    
    # O标签指标
    o_precision, o_recall, o_f1 = calculate_metrics(o_tp, o_fp, o_fn)
    print(f"O Label - Precision: {o_precision:.4f}, Recall: {o_recall:.4f}, F1: {o_f1:.4f}")
    print(f"         TP: {o_tp}, FP: {o_fp}, FN: {o_fn}")
    
    # 计算宏平均
    macro_precision = (b_precision + i_precision + o_precision) / 3
    macro_recall = (b_recall + i_recall + o_recall) / 3
    macro_f1 = b_f1*0.4 + i_f1*0.4 + o_f1*0.2
    print(f"\nMacro Average - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
                
                

def evaluate_type_task(model, dataloader, device, id2type_label):
    """评估类型识别任务"""
    all_predictions = []
    all_labels = []
    non_allusion_correct = 0
    non_allusion_total = 0
    allusion_correct = 0
    allusion_total = 0
    
    # 添加top-k指标统计
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Evaluating type task")
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            dict_features = {
                'indices': batch['dict_features']['indices'].to(device),
                'values': batch['dict_features']['values'].to(device),
                'active_counts': batch['dict_features']['active_counts'].to(device)
            }
            target_positions = batch['target_positions'].to(device)
            type_labels = batch['type_labels'].to(device)
            
            # 获取预测结果
            pred_top5 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_features=dict_features,
                task='type',
                target_positions=target_positions
            )
            
            predictions = pred_top5['predictions']  # 保留所有top-k预测
            labels = type_labels.squeeze()
            
            # 统计top-k准确率
            for pred_k, label in zip(predictions, labels):
                total_samples += 1
                if label in pred_k[:1]:  # top1
                    top1_correct += 1
                if label in pred_k[:3]:  # top3
                    top3_correct += 1
                if label in pred_k[:5]:  # top5
                    top5_correct += 1
            
            # 统计非典故和典故样本的准确率（使用top1预测）
            for pred, label in zip(predictions[:, 0], labels):
                if label == 0:  # 非典故样本
                    non_allusion_total += 1
                    if pred == 0:
                        non_allusion_correct += 1
                else:  # 典故样本
                    allusion_total += 1
                    if pred == label:
                        allusion_correct += 1
            
            all_predictions.extend(predictions[:, 0].cpu().numpy())  # 使用top1预测计算混淆矩阵
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条信息
            pbar.set_postfix({
                'top1_acc': f"{top1_correct/total_samples:.4f}",
                'top3_acc': f"{top3_correct/total_samples:.4f}",
                'top5_acc': f"{top5_correct/total_samples:.4f}"
            })
    
    
    # 计算指标
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(id2type_label.keys()),
        target_names=[id2type_label[i] for i in id2type_label],
        digits=4
    )
    

    
    # 计算各项准确率
    non_allusion_accuracy = non_allusion_correct / non_allusion_total if non_allusion_total > 0 else 0
    allusion_accuracy = allusion_correct / allusion_total if allusion_total > 0 else 0
    total_accuracy = (non_allusion_correct + allusion_correct) / (non_allusion_total + allusion_total)
    
    # 添加top-k准确率到返回结果
    accuracy_stats = {
        'non_allusion_accuracy': non_allusion_accuracy,
        'allusion_accuracy': allusion_accuracy,
        'total_accuracy': total_accuracy,
        'non_allusion_stats': f"{non_allusion_correct}/{non_allusion_total}",
        'allusion_stats': f"{allusion_correct}/{allusion_total}",
        'top1_accuracy': top1_correct / total_samples,
        'top3_accuracy': top3_correct / total_samples,
        'top5_accuracy': top5_correct / total_samples
    }
    
    return report, accuracy_stats

def save_evaluation_results(save_dir, position_report, type_report, accuracy_stats):
    """保存评估结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(save_dir, f'evaluation_results_{timestamp}.txt')
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("=== Position Recognition Results ===\n")
        f.write(position_report)
        f.write("\n\n=== Type Classification Results ===\n")
        f.write(type_report)
        f.write("\n\nAccuracy Statistics:\n")
        f.write(f"Non-allusion Accuracy: {accuracy_stats['non_allusion_accuracy']:.4f} ({accuracy_stats['non_allusion_stats']})\n")
        f.write(f"Allusion Accuracy: {accuracy_stats['allusion_accuracy']:.4f} ({accuracy_stats['allusion_stats']})\n")
        f.write(f"Total Accuracy: {accuracy_stats['total_accuracy']:.4f}\n")
    
    print(f"\nResults saved to: {result_file}")

def main():
    try:
        # 加载模型和数据
        # 加载典故词典和类型映射
        _, type_label2id, id2type_label, _ = load_allusion_dict()
        position_model, type_model, tokenizer, device = load_models()
        
        
        # 预处理特征和映射文件路径
        features_path = os.path.join(DATA_DIR, 'allusion_features_strictly_dict.pt')
        mapping_path = os.path.join(DATA_DIR, 'allusion_mapping_strictly_dict.json')
        # 评估位置识别任务
        print("\n=== Position Recognition Task ===")
        position_test_dataset = PoetryNERDataset(
            os.path.join(DATA_DIR, '4_test_position_no_bug.csv'),
            tokenizer, MAX_SEQ_LEN,
            type_label2id=type_label2id,
            id2type_label=id2type_label,
            task='position',
            features_path=features_path,
            mapping_path=mapping_path
        )
        
        test_position_dataloader = DataLoader(
            position_test_dataset, 
            batch_size=BATCH_SIZE,
            shuffle=False,  # 验证集不需要打乱
            collate_fn=position_test_dataset.collate_fn
        )
        
        # 评估前清理GPU内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print("\nGPU Memory Before Position Task:")
            print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        position_report = evaluate_position_task(position_model, test_position_dataloader, device)
        print(position_report)
        
        # 评估类型识别任务
        print("\n=== Type Classification Task ===")
        test_type_dataset = PoetryNERDataset(
            os.path.join(DATA_DIR, '4_test_type_no_bug.csv'),
            tokenizer, MAX_SEQ_LEN,
            type_label2id=type_label2id,
            id2type_label=id2type_label,
            task='type',
            features_path=features_path,
            mapping_path=mapping_path
        )
        
        test_type_dataloader = DataLoader(
            test_type_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,  # 验证集不需要打乱
            collate_fn=test_type_dataset.collate_fn
        )
        
        # 评估前清理GPU内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print("\nGPU Memory Before Type Task:")
            print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        type_report,accuracy_stats = evaluate_type_task(
            type_model, test_type_dataloader, device, id2type_label
        )
        print(type_report)
        print("\nAccuracy Statistics:")
        print(f"Non-allusion Accuracy: {accuracy_stats['non_allusion_accuracy']:.4f} ({accuracy_stats['non_allusion_stats']})")
        print(f"Allusion Accuracy: {accuracy_stats['allusion_accuracy']:.4f} ({accuracy_stats['allusion_stats']})")
        print(f"Total Accuracy: {accuracy_stats['total_accuracy']:.4f}")
        
        # 保存评估结果
        save_evaluation_results(SAVE_DIR, position_report, type_report, accuracy_stats)
        
        # 最终内存使用情况
        if device.type == 'cuda':
            print("\nFinal GPU Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            print(f"Cached: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
            
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()