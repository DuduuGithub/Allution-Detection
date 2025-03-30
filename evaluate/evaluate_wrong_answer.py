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
import json

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

def evaluate_jointly(model, dataloader, device, id2type_label, tokenizer):
    """联合评估位置识别和类型识别任务"""
    # 初始化统计变量
    total_loss = 0
    total_position_loss = 0
    total_type_loss = 0
    
    # 收集所有输出和标签用于评估
    all_outputs = []
    all_labels = []
    all_texts = []
    
    # 初始化错误案例收集列表
    position_errors = []
    type_errors = []
    all_errors = []
    
    pbar = tqdm(dataloader, desc="Evaluating jointly")
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_labels = batch['position_labels'].to(device)
            type_labels = batch['type_labels'].to(device)
            target_positions = batch['target_positions'].to(device)
            batch_texts = batch['text']
            
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
            
            # 收集当前批次的输出、标签和文本
            batch_outputs = {
                'position_predictions': position_pred_outputs['position_predictions'],
                'type_predictions': type_pred_outputs['type_predictions'],
                'attention_mask': attention_mask,
                'target_positions': target_positions
            }
            batch_labels = {
                'position_labels': position_labels,
                'type_labels': cleaned_type_labels,
                'attention_mask': attention_mask,
                'target_positions': target_positions
            }
            
            all_outputs.append(batch_outputs)
            all_labels.append(batch_labels)
            all_texts.append(batch_texts)
            
            # 对每个样本进行错误分析
            for idx in range(len(batch_texts)):
                text = batch_texts[idx]
                pos_pred = position_pred_outputs['position_predictions'][idx]
                pos_true = position_labels[idx]
                
                # 获取当前样本的类型预测
                sample_type_preds = type_pred_outputs['type_predictions'][idx]
                
                # 根据文本长度裁剪position标签
                text_tokens = tokenizer.tokenize(text)
                text_len = len(text_tokens)
                pos_true = pos_true[1:text_len+1].cpu().tolist()
                pos_pred = pos_pred[:text_len]
                
                # 确保使用张量进行比较
                has_position_error = not torch.equal(
                    torch.tensor(pos_pred),
                    torch.tensor(pos_true)
                )
                
                # 从预测元组中提取最可能的类别
                pred_type = sample_type_preds[2][0][0]  # 获取概率最高的类别ID
                current_type_label = cleaned_type_labels[idx]  # 使用清理后的标签
                has_type_error = pred_type != current_type_label.item()
                
                # 将预测和标签转换为列表以进行保存
                error_info = {
                    'text': text,
                    'position_pred': pos_pred,
                    'position_true': pos_true,
                    'type_pred': [id2type_label[sample_type_preds[2][0][0]]],  # 最可能的类别
                    'type_true': [id2type_label[current_type_label.item()]],  # 使用清理后的标签
                    'type_pred_probs': [(id2type_label[class_id], prob) for class_id, prob in sample_type_preds[2][:5]]
                }
                
                if has_position_error:
                    position_errors.append(error_info)
                if has_type_error:
                    type_errors.append(error_info)
                if has_position_error or has_type_error:
                    all_errors.append(error_info)
            
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
    
    return metrics, \
           {'total': avg_loss, 'position': avg_position_loss, 'type': avg_type_loss}, \
           {'position': position_errors, 'type': type_errors, 'all': all_errors}, \
           all_outputs, all_labels, all_texts

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

def save_error_cases(save_dir, error_cases, timestamp):
    """保存错误案例到文件"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存不同类型的错误
    for error_type, errors in error_cases.items():
        filename = os.path.join(save_dir, f'error_cases_{error_type}_{timestamp}.txt')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Total {error_type} errors: {len(errors)}\n\n")
            for i, error in enumerate(errors, 1):
                f.write(f"Error Case {i}:\n")
                f.write(f"Text: {error['text']}\n")
                f.write(f"Position Prediction: {error['position_pred']}\n")
                f.write(f"Position True: {error['position_true']}\n")
                f.write(f"Type Prediction: {error['type_pred']}\n")
                f.write(f"Type True: {error['type_true']}\n")
                f.write("\n" + "="*50 + "\n\n")
        print(f"Saved {error_type} error cases to: {filename}")

def save_error_cases_json(save_dir, error_type, errors, metrics, losses, timestamp):
    """保存单个错误类型的案例为JSON格式"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 构建结果字典
    results = {
        "timestamp": timestamp,
        "error_type": error_type,
        "total_errors": len(errors),
        "metrics": {
            "position": {
                "overall": {
                    "precision": metrics['position']['precision'],
                    "recall": metrics['position']['recall'],
                    "f1": metrics['position']['f1']
                },
                "by_label": {
                    label: {
                        "precision": metrics['position'][label]['precision'],
                        "recall": metrics['position'][label]['recall'],
                        "f1": metrics['position'][label]['f1'],
                        "tp": metrics['position'][label]['tp'],
                        "fp": metrics['position'][label]['fp'],
                        "fn": metrics['position'][label]['fn']
                    } for label in ['B', 'I', 'O']
                }
            },
            "type": {
                "top1_accuracy": metrics['type']['top1_acc'],
                "top3_accuracy": metrics['type']['top3_acc'],
                "top5_accuracy": metrics['type']['top5_acc'],
                "negative_accuracy": metrics['type']['negative_acc']
            }
        },
        "losses": {
            "total": float(losses['total']),
            "position": float(losses['position']),
            "type": float(losses['type'])
        },
        "error_cases": [
            {
                "text": error['text'],
                "position_pred": error['position_pred'],
                "position_true": error['position_true'],
                "type_pred": error['type_pred'],
                "type_true": error['type_true'],
                "type_pred_probs": [
                    {"type": t, "probability": float(p)} 
                    for t, p in error['type_pred_probs']
                ]
            } for error in errors
        ]
    }
    
    # 保存为JSON文件
    result_file = os.path.join(save_dir, f'error_cases_{error_type}_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {error_type} error cases to JSON: {result_file}")

def save_evaluation_results_json(save_dir, metrics, losses, error_cases, timestamp):
    """保存评估结果为JSON格式"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 构建结果字典
    results = {
        "timestamp": timestamp,
        "metrics": {
            "position": {
                "overall": {
                    "precision": metrics['position']['precision'],
                    "recall": metrics['position']['recall'],
                    "f1": metrics['position']['f1']
                },
                "by_label": {
                    label: {
                        "precision": metrics['position'][label]['precision'],
                        "recall": metrics['position'][label]['recall'],
                        "f1": metrics['position'][label]['f1'],
                        "tp": metrics['position'][label]['tp'],
                        "fp": metrics['position'][label]['fp'],
                        "fn": metrics['position'][label]['fn']
                    } for label in ['B', 'I', 'O']
                }
            },
            "type": {
                "top1_accuracy": metrics['type']['top1_acc'],
                "top3_accuracy": metrics['type']['top3_acc'],
                "top5_accuracy": metrics['type']['top5_acc'],
                "negative_accuracy": metrics['type']['negative_acc']
            }
        },
        "losses": {
            "total": float(losses['total']),
            "position": float(losses['position']),
            "type": float(losses['type'])
        },
        "error_cases": {
            "position": [
                {
                    "text": error['text'],
                    "position_pred": error['position_pred'],
                    "position_true": error['position_true'],
                    "type_pred": error['type_pred'],
                    "type_true": error['type_true'],
                    "type_pred_probs": [
                        {"type": t, "probability": float(p)} 
                        for t, p in error['type_pred_probs']
                    ]
                } for error in error_cases['position']
            ],
            "type": [
                {
                    "text": error['text'],
                    "position_pred": error['position_pred'],
                    "position_true": error['position_true'],
                    "type_pred": error['type_pred'],
                    "type_true": error['type_true'],
                    "type_pred_probs": [
                        {"type": t, "probability": float(p)} 
                        for t, p in error['type_pred_probs']
                    ]
                } for error in error_cases['type']
            ],
            "all": [
                {
                    "text": error['text'],
                    "position_pred": error['position_pred'],
                    "position_true": error['position_true'],
                    "type_pred": error['type_pred'],
                    "type_true": error['type_true'],
                    "type_pred_probs": [
                        {"type": t, "probability": float(p)} 
                        for t, p in error['type_pred_probs']
                    ]
                } for error in error_cases['all']
            ]
        }
    }
    
    # 保存为JSON文件
    result_file = os.path.join(save_dir, f'evaluation_results_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {result_file}")

def analyze_allusion_counts(error_cases, all_predictions, all_labels):
    """分析真实典故和预测典故的数量关系"""
    # 初始化计数器
    statistics = {
        "total_samples": 0,
        "true_allusions": 0,
        "predicted_allusions": 0,
        "samples_with_true_allusions": 0,
        "samples_with_predicted_allusions": 0,
        "count_comparison": {
            "pred_less": 0,    # 预测数量少于真实数量
            "pred_equal": 0,   # 预测数量等于真实数量
            "pred_more": 0     # 预测数量多于真实数量
        },
        "detailed_counts": []  # 存储每个样本的详细信息
    }
    
    def count_allusions(sequence):
        """计算序列中典故的数量（通过B标签计数）"""
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().tolist()
        return sum(1 for i in range(len(sequence)) if sequence[i] == 1)
    
    # 遍历所有样本
    for batch_outputs, batch_labels in zip(all_predictions, all_labels):
        batch_size = len(batch_outputs['position_predictions'])
        for idx in range(batch_size):
            # 获取当前样本的位置标签和预测
            true_positions = batch_labels['position_labels'][idx]
            pred_positions = batch_outputs['position_predictions'][idx]
            
            # 计算真实典故和预测典故的数量
            true_count = count_allusions(true_positions)
            pred_count = count_allusions(pred_positions)
            
            # 更新统计信息
            statistics["total_samples"] += 1
            statistics["true_allusions"] += true_count
            statistics["predicted_allusions"] += pred_count
            
            if true_count <=1:
                continue
            
            if true_count > 1:
                statistics["samples_with_true_allusions"] += 1
            if pred_count > 1:
                statistics["samples_with_predicted_allusions"] += 1
            
            # 比较预测和真实数量
            if pred_count < true_count:
                statistics["count_comparison"]["pred_less"] += 1
            elif pred_count == true_count:
                statistics["count_comparison"]["pred_equal"] += 1
            else:
                statistics["count_comparison"]["pred_more"] += 1
            
            # 记录详细信息
            statistics["detailed_counts"].append({
                "true_count": true_count,
                "pred_count": pred_count,
                "difference": pred_count - true_count
            })
    
    # 计算平均值和比率
    statistics["avg_true_per_sample"] = statistics["true_allusions"] / statistics["total_samples"]
    statistics["avg_pred_per_sample"] = statistics["predicted_allusions"] / statistics["total_samples"]
    statistics["true_allusion_sample_ratio"] = statistics["samples_with_true_allusions"] / statistics["total_samples"]
    statistics["pred_allusion_sample_ratio"] = statistics["samples_with_predicted_allusions"] / statistics["total_samples"]
    
    return statistics

def save_allusion_analysis(save_dir, statistics, timestamp):
    """保存典故数量分析结果"""
    analysis_file = os.path.join(save_dir, f'allusion_count_analysis_{timestamp}.txt')
    analysis_json = os.path.join(save_dir, f'allusion_count_analysis_{timestamp}.json')
    
    # 保存文本格式
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("=== Allusion Count Analysis ===\n\n")
        f.write(f"Total Samples: {statistics['total_samples']}\n")
        f.write(f"Total True Allusions: {statistics['true_allusions']}\n")
        f.write(f"Total Predicted Allusions: {statistics['predicted_allusions']}\n\n")
        
        f.write("Average Counts:\n")
        f.write(f"  True Allusions per Sample: {statistics['avg_true_per_sample']:.2f}\n")
        f.write(f"  Predicted Allusions per Sample: {statistics['avg_pred_per_sample']:.2f}\n\n")
        
        f.write("Sample Statistics:\n")
        f.write(f"  Samples with True Allusions: {statistics['samples_with_true_allusions']} ({statistics['true_allusion_sample_ratio']:.2%})\n")
        f.write(f"  Samples with Predicted Allusions: {statistics['samples_with_predicted_allusions']} ({statistics['pred_allusion_sample_ratio']:.2%})\n\n")
        
        f.write("Count Comparison:\n")
        f.write(f"  Predictions < Truth: {statistics['count_comparison']['pred_less']} ({statistics['count_comparison']['pred_less']/statistics['total_samples']:.2%})\n")
        f.write(f"  Predictions = Truth: {statistics['count_comparison']['pred_equal']} ({statistics['count_comparison']['pred_equal']/statistics['total_samples']:.2%})\n")
        f.write(f"  Predictions > Truth: {statistics['count_comparison']['pred_more']} ({statistics['count_comparison']['pred_more']/statistics['total_samples']:.2%})\n")
    
    # 保存JSON格式
    with open(analysis_json, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    
    print(f"\nAllusion count analysis saved to: {analysis_file}")
    print(f"Detailed JSON data saved to: {analysis_json}")

def save_count_mismatch_samples(save_dir, all_outputs, all_labels, all_texts, timestamp):
    """保存预测数量与真实数量不匹配的样本"""
    def count_allusions(sequence):
        """计算序列中典故的数量（通过B标签计数）"""
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().tolist()
        return sum(1 for i in range(len(sequence)) if sequence[i] == 1)
    
    # 创建两个文件
    less_file = os.path.join(save_dir, f'pred_less_than_true_{timestamp}.txt')
    more_file = os.path.join(save_dir, f'pred_more_than_true_{timestamp}.txt')
    
    with open(less_file, 'w', encoding='utf-8') as f_less, \
         open(more_file, 'w', encoding='utf-8') as f_more:
        
        f_less.write("=== Samples where predicted allusions < true allusions ===\n\n")
        f_more.write("=== Samples where predicted allusions > true allusions ===\n\n")
        
        sample_idx = 0
        for batch_outputs, batch_labels, batch_texts in zip(all_outputs, all_labels, all_texts):
            batch_size = len(batch_outputs['position_predictions'])
            
            for idx in range(batch_size):
                # 获取当前样本的位置标签和预测
                true_positions = batch_labels['position_labels'][idx]
                pred_positions = batch_outputs['position_predictions'][idx]
                text = batch_texts[idx]
                
                # 计算真实典故和预测典故的数量
                true_count = count_allusions(true_positions)
                pred_count = count_allusions(pred_positions)
                
                # 跳过 true_count 或 pred_count 为 0 的情况
                if true_count == 0 or pred_count == 0:
                    continue
                
                # 根据数量差异写入相应文件
                if pred_count < true_count:
                    f_less.write(f"Sample {sample_idx + 1}:\n")
                    f_less.write(f"Text: {text}\n")
                    f_less.write(f"True positions: {true_positions}\n")
                    f_less.write(f"Pred positions: {pred_positions}\n")
                    f_less.write(f"True count: {true_count}, Pred count: {pred_count}\n")
                    f_less.write("="*50 + "\n\n")
                elif pred_count > true_count:
                    f_more.write(f"Sample {sample_idx + 1}:\n")
                    f_more.write(f"Text: {text}\n")
                    f_more.write(f"True positions: {true_positions}\n")
                    f_more.write(f"Pred positions: {pred_positions}\n")
                    f_more.write(f"True count: {true_count}, Pred count: {pred_count}\n")
                    f_more.write("="*50 + "\n\n")
                
                sample_idx += 1
    
    print(f"\nSamples with fewer predictions saved to: {less_file}")
    print(f"Samples with more predictions saved to: {more_file}")

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
        metrics, losses, error_cases, all_outputs, all_labels, all_texts = evaluate_jointly(
            model, test_dataloader, device, id2type_label, tokenizer
        )
        
        # 分析典故数量
        statistics = analyze_allusion_counts(error_cases, all_outputs, all_labels)
        
        # 使用os.path来构建正确的保存路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, 'analysis_results')
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存典故数量分析结果
        save_allusion_analysis(save_dir, statistics, timestamp)
        
        # 使用收集到的文本
        save_count_mismatch_samples(save_dir, all_outputs, all_labels, all_texts, timestamp)
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise

if __name__ == '__main__':
    main()