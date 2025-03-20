import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import BertTokenizer
from model.poetry_dataset import PoetryNERDataset
from model.bert_crf import AllusionBERTCRF
from model.train import load_allusion_dict
from model.config import (
    BERT_MODEL_PATH, MAX_SEQ_LEN,  
    BATCH_SIZE, DATA_DIR
)
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from model.train import evaluate_metrics_from_outputs

def load_models(model_name):
    """加载预训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    # 加载典故词典以获取类型数量
    allusion_dict, _, _, num_types = load_allusion_dict()
    dict_size = len(allusion_dict)
    
    # 创建一个模型实例
    model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size=dict_size).to(device)
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAVE_DIR = os.path.join(PROJECT_ROOT, 'trained_result')
    
    # 加载模型参数并处理多余的键
    position_checkpoint = torch.load(f'{SAVE_DIR}/{model_name}.pt', map_location=device)
    
    print('testing model path:',f'{SAVE_DIR}/{model_name}.pt')
    
    state_dict = position_checkpoint['model_state_dict']
    
    # 移除多余的键
    for key in list(state_dict.keys()):
        if key not in model.state_dict():
            print(f"Removing unexpected key: {key}")
            del state_dict[key]
    
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, device

def evaluate_jointly(model, dataloader, device, id2type_label):
    """联合评估位置识别和类型识别任务"""
    # 初始化统计变量
    total_loss = 0
    total_position_loss = 0
    total_type_loss = 0
    
    # 收集所有输出和标签用于评估
    all_outputs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Evaluating jointly")
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_labels = batch['position_labels'].to(device)
            target_positions = batch['target_positions'].to(device)
            type_labels = batch['type_labels'].to(device)
            
            dict_features = {
                'indices': batch['dict_features']['indices'].to(device),
                'values': batch['dict_features']['values'].to(device),
                'active_counts': batch['dict_features']['active_counts'].to(device)
            }
            
            # 1. 计算损失（使用所有标签）
            loss_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_features=dict_features,
                position_labels=position_labels,
                target_positions=target_positions,
                type_labels=type_labels,
                train_mode=True
            )
            
            # 累计损失
            total_loss += loss_outputs['loss'].item()
            total_position_loss += loss_outputs['position_loss']
            total_type_loss += loss_outputs['type_loss']
            
            # 2. 获取预测结果
            position_pred_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_features=dict_features,
                train_mode=False,
                task='position'
            )
            
            type_pred_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_features=dict_features,
                target_positions=target_positions,
                train_mode=False,
                task='type'
            )
            
            # 清理type_labels
            cleaned_type_labels = []
            batch_size = type_labels.size(0)
            max_type_len = type_labels.size(1)
            for batch_idx in range(batch_size):
                for type_idx in range(max_type_len):
                    if target_positions[batch_idx][type_idx].sum() > 0:
                        cleaned_type_labels.append(type_labels[batch_idx][type_idx])
            
            all_outputs.append({
                'position_predictions': position_pred_outputs['position_predictions'],
                'type_predictions': type_pred_outputs['type_predictions']
            })
            
            all_labels.append({
                'position_labels': position_labels,
                'type_labels': cleaned_type_labels,
                'attention_mask': attention_mask
            })
            
            # 更新进度条
            avg_loss = total_loss / (pbar.n + 1)
            pbar.set_postfix({'avg_loss': f"{avg_loss:.4f}"})
    
    # 计算平均损失
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_position_loss = total_position_loss / num_batches
    avg_type_loss = total_type_loss / num_batches
    
    # 使用train.py中的evaluate_metrics_from_outputs计算评估指标
    metrics = evaluate_metrics_from_outputs(all_outputs, all_labels)
    
    # 打印详细评估结果
    print("\n=== Joint Evaluation Results ===")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Position Loss: {avg_position_loss:.4f}")
    print(f"Type Loss: {avg_type_loss:.4f}")
    
    print("\nPosition Recognition:")
    print(f"  Overall:")
    print(f"    Precision: {metrics['position']['precision']:.4f}")
    print(f"    Recall: {metrics['position']['recall']:.4f}")
    print(f"    F1: {metrics['position']['f1']:.4f}")
    
    for label in ['B', 'I', 'O']:
        print(f"  {label} Label:")
        print(f"    Precision: {metrics['position'][label]['precision']:.4f}")
        print(f"    Recall: {metrics['position'][label]['recall']:.4f}")
        print(f"    F1: {metrics['position'][label]['f1']:.4f}")
        print(f"    TP: {metrics['position'][label]['tp']}, FP: {metrics['position'][label]['fp']}, FN: {metrics['position'][label]['fn']}")
    
    print("\nType Recognition:")
    print(f"  Top-1 Accuracy: {metrics['type']['top1_acc']:.4f}")
    print(f"  Top-3 Accuracy: {metrics['type']['top3_acc']:.4f}")
    print(f"  Top-5 Accuracy: {metrics['type']['top5_acc']:.4f}")
    print(f"  Negative Accuracy: {metrics['type']['negative_acc']:.4f}")
    
    return metrics, {'total': avg_loss, 'position': avg_position_loss, 'type': avg_type_loss}

def save_evaluation_results(save_dir, metrics, losses):
    """保存评估结果到文件"""
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(save_dir, f'evaluation_results_{timestamp}.txt')
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("=== Joint Evaluation Results ===\n")
        f.write(f"Average Loss: {losses['total']:.4f}\n")
        f.write(f"Position Loss: {losses['position']:.4f}\n")
        f.write(f"Type Loss: {losses['type']:.4f}\n")
        
        f.write("\nPosition Recognition:\n")
        f.write(f"  Overall:\n")
        f.write(f"    Precision: {metrics['position']['precision']:.4f}\n")
        f.write(f"    Recall: {metrics['position']['recall']:.4f}\n")
        f.write(f"    F1: {metrics['position']['f1']:.4f}\n")
        
        for label in ['B', 'I', 'O']:
            f.write(f"  {label} Label:\n")
            f.write(f"    Precision: {metrics['position'][label]['precision']:.4f}\n")
            f.write(f"    Recall: {metrics['position'][label]['recall']:.4f}\n")
            f.write(f"    F1: {metrics['position'][label]['f1']:.4f}\n")
            f.write(f"    TP: {metrics['position'][label]['tp']}, FP: {metrics['position'][label]['fp']}, FN: {metrics['position'][label]['fn']}\n")
        
        f.write("\nType Recognition:\n")
        f.write(f"  Top-1 Accuracy: {metrics['type']['top1_acc']:.4f}\n")
        f.write(f"  Top-3 Accuracy: {metrics['type']['top3_acc']:.4f}\n")
        f.write(f"  Top-5 Accuracy: {metrics['type']['top5_acc']:.4f}\n")
        f.write(f"  Negative Accuracy: {metrics['type']['negative_acc']:.4f}\n")
    
    print(f"\nResults saved to: {result_file}")

def main():
    try:
        # 加载模型和数据
        _, type_label2id, id2type_label, _ = load_allusion_dict()
        model, tokenizer, device = load_models('output_jointly_train_normalize_loss/best_model_3.17.22.09')
        
        # 预处理特征和映射文件路径
        features_path = os.path.join(DATA_DIR, 'allusion_features_strictly_dict.pt')
        mapping_path = os.path.join(DATA_DIR, 'allusion_mapping_strictly_dict.json')
        
        # 创建测试数据集
        test_dataset = PoetryNERDataset(
            os.path.join(DATA_DIR, '4_test_position_no_bug_less_negatives.csv'),
            tokenizer, MAX_SEQ_LEN,
            type_label2id=type_label2id,
            id2type_label=id2type_label,
            features_path=features_path,
            mapping_path=mapping_path,
            negative_sample_ratio=0.05
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=test_dataset.collate_fn
        )
        
        # 评估前清理GPU内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print("\nGPU Memory Before Evaluation:")
            print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        # 联合评估
        metrics, losses = evaluate_jointly(model, test_dataloader, device, id2type_label)
        
        # 使用os.path来构建正确的保存路径
        current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
        save_dir = os.path.join(current_dir, 'evaluate_output_jointly')
        
        # 保存评估结果
        save_evaluation_results(save_dir, metrics, losses)
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()