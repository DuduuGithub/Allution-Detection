import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import BertTokenizer
from model.poetry_dataset import PoetryNERDataset
from model.bert_crf import AllusionBERTCRF, prepare_sparse_features
from model.train import load_allusion_dict
from model.config import (
    BERT_MODEL_PATH, MAX_SEQ_LEN, SAVE_DIR, 
    BATCH_SIZE, Test_DIR
)
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_models():
    """加载预训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    # 加载典故词典以获取类型数量
    allusion_dict, type_label2id, id2type_label, num_types = load_allusion_dict()
    dict_size = len(allusion_dict)
    
    # 创建两个模型实例
    position_model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size=dict_size).to(device)
    type_model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size=dict_size).to(device)
    
    # 加载模型参数
    position_checkpoint = torch.load(f'{Test_DIR}/best_model_position.pt', map_location=device)
    position_model.load_state_dict(position_checkpoint['model_state_dict'])
    position_model.eval()
    
    type_checkpoint = torch.load(f'{Test_DIR}/best_model_type.pt', map_location=device)
    type_model.load_state_dict(type_checkpoint['model_state_dict'])
    type_model.eval()
    
    return position_model, type_model, tokenizer, device, type_label2id, id2type_label

def evaluate_position_task(model, dataloader, device):
    """评估位置识别任务"""
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            dict_features = batch['dict_features']
            
            if dict_features is not None:
                dict_features = {k: v.to(device) for k, v in dict_features.items()}
            
            # 获取预测结果
            predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_features=dict_features,
                task='position'
            )
            
            # 获取真实标签
            labels = batch['position_labels'].cpu().numpy()
            
            # 收集预测结果和标签
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
    # 计算指标
    report = classification_report(
        [label for labels in all_labels for label in labels],
        [pred for preds in all_predictions for pred in preds],
        labels=[0, 1, 2],
        target_names=['O', 'B', 'I'],
        digits=4
    )
    
    return report

def analyze_confusion_matrix(cm, id2type_label, save_dir):
    """分析混淆矩阵并生成多个有意义的可视化和统计
    
    Args:
        cm: 完整的混淆矩阵
        id2type_label: 类型ID到标签的映射
        save_dir: 保存结果的目录
    """
    n_classes = len(id2type_label)
    
    # 1. 计算每个类别的主要混淆情况
    class_confusion = {}
    for true_id in range(n_classes):
        if true_id == 0:  # 跳过非典故类型
            continue
            
        true_label = id2type_label[true_id]
        total_samples = cm[true_id].sum()
        if total_samples == 0:
            continue
            
        # 获取前5个最容易混淆的类别
        confusion_scores = []
        for pred_id in range(n_classes):
            if pred_id != true_id and cm[true_id, pred_id] > 0:
                confusion_scores.append((pred_id, cm[true_id, pred_id]))
        
        top_confusions = sorted(confusion_scores, key=lambda x: x[1], reverse=True)[:5]
        
        class_confusion[true_label] = {
            'total_samples': total_samples,
            'correct_predictions': cm[true_id, true_id],
            'accuracy': cm[true_id, true_id] / total_samples,
            'top_confusions': [(id2type_label[pid], count) for pid, count in top_confusions]
        }
    
    # 2. 生成最严重的混淆子矩阵（Top 20）
    # 选择混淆最严重的20个类别
    class_errors = []
    for i in range(n_classes):
        if i == 0:  # 跳过非典故类型
            continue
        total_errors = cm[i].sum() - cm[i,i]  # 总错误数
        if total_errors > 0:
            class_errors.append((i, total_errors))
    
    top_confused_ids = [i for i, _ in sorted(class_errors, key=lambda x: x[1], reverse=True)[:20]]
    top_confused_labels = [id2type_label[i] for i in top_confused_ids]
    
    # 绘制Top 20混淆子矩阵
    sub_cm = cm[top_confused_ids][:, top_confused_ids]
    plt.figure(figsize=(15, 12))
    sns.heatmap(sub_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=top_confused_labels,
                yticklabels=top_confused_labels)
    plt.title('Top 20 Most Confused Classes')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'top20_confusion_matrix.png'))
    plt.close()
    
    # 3. 生成统计报告
    report = ["=== 典故识别混淆分析报告 ===\n"]
    
    # 3.1 总体统计
    total_samples = cm.sum()
    correct_predictions = cm.diagonal().sum()
    report.append(f"总样本数: {total_samples}")
    report.append(f"总体准确率: {correct_predictions/total_samples:.4f}\n")
    
    # 3.2 非典故类别统计
    non_allusion_accuracy = cm[0,0] / cm[0].sum() if cm[0].sum() > 0 else 0
    report.append(f"非典故样本准确率: {non_allusion_accuracy:.4f}")
    report.append(f"非典故样本数量: {cm[0].sum()}")
    report.append(f"非典故正确识别: {cm[0,0]}\n")
    
    # 3.3 典故混淆统计
    report.append("=== 典故混淆统计（Top 20）===")
    for true_label in top_confused_labels:
        stats = class_confusion[true_label]
        report.append(f"\n典故: {true_label}")
        report.append(f"样本数: {stats['total_samples']}")
        report.append(f"准确率: {stats['accuracy']:.4f}")
        report.append("主要混淆:")
        for confused_label, count in stats['top_confusions']:
            report.append(f"  - 误识别为 {confused_label}: {count}次")
    
    # 保存报告
    with open(os.path.join(save_dir, 'confusion_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return class_confusion

def evaluate_type_task(model, dataloader, device, id2type_label):
    """评估类型识别任务"""
    all_predictions = []
    all_labels = []
    non_allusion_correct = 0
    non_allusion_total = 0
    allusion_correct = 0
    allusion_total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            dict_features = batch['dict_features']
            target_positions = batch['target_positions'].to(device)
            type_labels = batch['type_labels'].to(device)
            
            if dict_features is not None:
                dict_features = {k: v.to(device) for k, v in dict_features.items()}
            
            # 获取预测结果
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_features=dict_features,
                task='type',
                target_positions=target_positions
            )
            
            predictions = output['predictions'][:, 0]  # 取top1预测
            labels = type_labels.squeeze()
            
            # 统计非典故和典故样本的准确率
            for pred, label in zip(predictions, labels):
                if label == 0:  # 非典故样本
                    non_allusion_total += 1
                    if pred == 0:
                        non_allusion_correct += 1
                else:  # 典故样本
                    allusion_total += 1
                    if pred == label:
                        allusion_correct += 1
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(id2type_label.keys()),
        target_names=[id2type_label[i] for i in id2type_label],
        digits=4
    )
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 分析混淆矩阵
    class_confusion = analyze_confusion_matrix(cm, id2type_label, Test_DIR)
    
    # 计算各项准确率
    non_allusion_accuracy = non_allusion_correct / non_allusion_total if non_allusion_total > 0 else 0
    allusion_accuracy = allusion_correct / allusion_total if allusion_total > 0 else 0
    total_accuracy = (non_allusion_correct + allusion_correct) / (non_allusion_total + allusion_total)
    
    return report, cm, {
        'non_allusion_accuracy': non_allusion_accuracy,
        'allusion_accuracy': allusion_accuracy,
        'total_accuracy': total_accuracy,
        'non_allusion_stats': f"{non_allusion_correct}/{non_allusion_total}",
        'allusion_stats': f"{allusion_correct}/{allusion_total}",
        'class_confusion': class_confusion
    }

def plot_confusion_matrix(cm, labels, save_path):
    """绘制并保存混淆矩阵热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 加载模型和数据
    position_model, type_model, tokenizer, device, type_label2id, id2type_label = load_models()
    
    # 准备测试数据
    test_file = os.path.join(os.path.dirname(Test_DIR), 'data', '4_test_position_no_bug.csv')
    
    # 评估位置识别任务
    test_position_dataset = PoetryNERDataset(
        test_file, tokenizer, MAX_SEQ_LEN,
        type_label2id=type_label2id,
        id2type_label=id2type_label,
        task='position'
    )
    
    test_position_dataloader = DataLoader(
        test_position_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=test_position_dataset.collate_fn
    )
    
    print("\n=== Position Recognition Task ===")
    position_report = evaluate_position_task(position_model, test_position_dataloader, device)
    print(position_report)
    
    # 评估类型识别任务
    test_type_dataset = PoetryNERDataset(
        test_file, tokenizer, MAX_SEQ_LEN,
        type_label2id=type_label2id,
        id2type_label=id2type_label,
        task='type'
    )
    
    test_type_dataloader = DataLoader(
        test_type_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=test_type_dataset.collate_fn
    )
    
    print("\n=== Type Classification Task ===")
    type_report, confusion_mat, accuracy_stats = evaluate_type_task(
        type_model, test_type_dataloader, device, id2type_label
    )
    print(type_report)
    print("\nAccuracy Statistics:")
    print(f"Non-allusion Accuracy: {accuracy_stats['non_allusion_accuracy']:.4f} ({accuracy_stats['non_allusion_stats']})")
    print(f"Allusion Accuracy: {accuracy_stats['allusion_accuracy']:.4f} ({accuracy_stats['allusion_stats']})")
    print(f"Total Accuracy: {accuracy_stats['total_accuracy']:.4f}")
    
    # 保存混淆矩阵图
    plot_confusion_matrix(
        confusion_mat,
        [id2type_label[i] for i in range(len(id2type_label))],
        os.path.join(Test_DIR, 'confusion_matrix.png')
    )

if __name__ == "__main__":
    main()