import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.poetry_dataset import PoetryNERDataset
from model.bert_crf import AllusionBERTCRF, prepare_sparse_features
from model.config import (
    BERT_MODEL_PATH, MAX_SEQ_LEN, BATCH_SIZE, 
    EPOCHS,
    SAVE_DIR, DATA_DIR, ALLUSION_DICT_PATH
)
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import argparse
import pandas as pd
from datetime import datetime
import math


# # 定义加权得分计算函数
# def calculate_weighted_score(metrics, w1=0.4, w2=0.4, w3=0.2):
#     """
#     计算加权得分
#     :param metrics: 评估指标字典
#     :param w1: B/I 标签准确率的权重
#     :param w2: B/I 标签召回率的权重
#     :param w3: 类别识别任务准确率的权重
#     :return: 加权得分
#     """
#     precision_bi = (metrics["position"]["B"]["precision"] + metrics["position"]["I"]["precision"]) / 2
#     recall_bi = (metrics["position"]["B"]["recall"] + metrics["position"]["I"]["recall"]) / 2
#     top1_accuracy = metrics["type"]["top1_acc"]
#     weighted_score = w1 * precision_bi + w2 * recall_bi + w3 * top1_accuracy
#     return weighted_score



def load_allusion_dict(dict_file=ALLUSION_DICT_PATH):
    """加载典故词典并创建类型映射
    
    Returns:
        tuple: (
            allusion_dict: Dict[str, List[str]],  # {典故代表词: [变体列表]}
            type_label2id: Dict[str, int],        # {类型名: 类型ID}
            id2type_label: Dict[int, str],        # {类型ID: 类型名}
            num_types: int                        # 类型总数（包含非典故类型）
        )
    """
    # 1. 读取典故词典
    allusion_dict = {}
    type_set = set()  # 用于收集所有类型
    
    df = pd.read_csv(dict_file, encoding='utf-8', sep='\t')
    for _, row in df.iterrows():
        # 处理典故名和类型
        allusion = row['allusion']
    
        # 处理变体列表
        variants = eval(row['variation_list'])  # 安全地解析字符串列表
        
        # 添加到典故词典
        allusion_dict[allusion] = variants
        # 收集类型
        type_set.add(allusion)
    
    # 2. 创建类型映射
    # 首先添加非典故类型（ID为0）
    type_label2id = {'O': 0}          # 'O' 表示非典故
    id2type_label = {0: 'O'}
    
    # 对其他类型名称排序以确保映射的一致性
    sorted_types = sorted(list(type_set))
    for idx, label in enumerate(sorted_types, start=1):  # 从1开始编号
        type_label2id[label] = idx
        id2type_label[idx] = label
    
    num_types = len(type_label2id)  # 包含非典故类型
    
    print(f"Loaded allusion dictionary with {len(allusion_dict)} entries")
    print(f"Found {num_types} types (including non-allusion type)")
    print(f"Type label 0 is reserved for non-allusion")
    
    return allusion_dict, type_label2id, id2type_label, num_types

def calculate_metrics(position_correct, position_pred_total, position_true_total,
                     type_correct, type_pred_total, type_true_total,
                     type_top3_correct, type_top5_correct):
    """计算评估指标
    
    Args:
        position_correct (int): 位置预测正确的数量
        position_pred_total (int): 位置预测的总数量
        position_true_total (int): 位置真实的总数量
        type_correct (int): 类型预测正确的数量
        type_pred_total (int): 类型预测的总数量
        type_true_total (int): 类型真实的总数量
        type_top3_correct (int): Top-3正确的数量
        type_top5_correct (int): Top-5正确的数量
    
    Returns:
        dict: 包含各项评估指标的字典
    """
    # 计算位置识别的指标
    position_precision = position_correct / position_pred_total if position_pred_total > 0 else 0
    position_recall = position_correct / position_true_total if position_true_total > 0 else 0
    position_f1 = 2 * (position_precision * position_recall) / (position_precision + position_recall) \
                 if (position_precision + position_recall) > 0 else 0

    # 计算类型识别的指标
    type_precision = type_correct / type_pred_total if type_pred_total > 0 else 0
    type_recall = type_correct / type_true_total if type_true_total > 0 else 0
    type_f1 = 2 * (type_precision * type_recall) / (type_precision + type_recall) \
              if (type_precision + type_recall) > 0 else 0
    
    # 计算Top-k准确率
    type_top3_acc = type_top3_correct / type_true_total if type_true_total > 0 else 0
    type_top5_acc = type_top5_correct / type_true_total if type_true_total > 0 else 0

    return {
        'position': {
            'precision': position_precision,
            'recall': position_recall,
            'f1': position_f1
        },
        'type': {
            'precision': type_precision,
            'recall': type_recall,
            'f1': type_f1,
            'top3_acc': type_top3_acc,
            'top5_acc': type_top5_acc
        }
    }

def evaluate_metrics_from_outputs(outputs, labels):
    """使用已有的预测结果和标签计算评估指标"""
    position_correct = 0
    position_pred_total = 0
    position_true_total = 0
    
    # B/I/O标签的统计
    b_tp, b_fp, b_fn = 0, 0, 0  # B标签的true positive, false positive, false negative
    i_tp, i_fp, i_fn = 0, 0, 0  # I标签的统计
    o_tp, o_fp, o_fn = 0, 0, 0  # O标签的统计
    
    #类别统计
    type_positive_correct = 0
    type_positive_total = 0
    type_positive_top3_correct = 0
    type_positive_top5_correct = 0
    
    # 添加负例统计
    type_negative_total = 0      # 总负例数
    type_negative_correct = 0    # 正确预测的负例数
    
    # 错误统计
    type_mistake_positive_to_negative = 0
    type_mistake_negative_to_positive = 0
    
    for batch_outputs, batch_labels in zip(outputs, labels):
        
        # 位置识别统计
        pred_positions = batch_outputs['position_predictions']
        true_positions = batch_labels['position_labels']
        attention_mask = batch_labels['attention_mask']
        
        for pred, true, mask in zip(pred_positions, true_positions, attention_mask):
            seq_len = mask.sum().item() - 2
            true = true[1:seq_len+1]
            
            # 统计B/I/O标签
            for p, t in zip(pred, true):
                # B标签统计
                if t == 1:  # 真实标签是B
                    if p == 1: b_tp += 1
                    else: b_fn += 1
                if p == 1:  # 预测标签是B
                    if t != 1: b_fp += 1
                
                # I标签统计
                if t == 2:  # 真实标签是I
                    if p == 2: i_tp += 1
                    else: i_fn += 1
                if p == 2:  # 预测标签是I
                    if t != 2: i_fp += 1
                
                # O标签统计
                if t == 0:  # 真实标签是O
                    if p == 0: o_tp += 1
                    else: o_fn += 1
                if p == 0:  # 预测标签是O
                    if t != 0: o_fp += 1
            
            # 原有的位置统计
            position_pred_total += sum(1 for p in pred if p > 0)
            position_true_total += sum(1 for t in true if t > 0)
            position_correct += sum(1 for p, t in zip(pred, true) if p == t and p > 0)
        
        # 类型识别统计
        type_labels = batch_labels['type_labels']
        type_predictions = batch_outputs['type_predictions']
        
        # 确保预测结果和标签数量匹配
        for pred_info,type_label in zip(type_predictions,type_labels):
            _,_,pred_types = pred_info  # 解包预测结果
            # 获取预测的类型（top1-5）
            pred_type_ids = [p[0] for p in pred_types]
            
            #正例
            if type_label > 0:  
                type_positive_total += 1
                # 检查top1准确率
                if pred_type_ids[0] == type_label:
                    type_positive_correct += 1
                    
                # 检查top3准确率
                if type_label in pred_type_ids[:3]:
                    type_positive_top3_correct += 1
                    
                # 检查top5准确率
                if type_label in pred_type_ids[:5]:
                    type_positive_top5_correct += 1
                
                else:
                    type_mistake_positive_to_negative += 1
            
            #负例
            else:
                type_negative_total += 1
                if pred_type_ids[0] == 0:  # 正确预测为非典故
                    type_negative_correct += 1
                else:
                    type_mistake_negative_to_positive += 1
    
    # 计算评估指标
    def calculate_label_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1
    
    # 计算B/I/O标签的指标
    b_precision, b_recall, b_f1 = calculate_label_metrics(b_tp, b_fp, b_fn)
    i_precision, i_recall, i_f1 = calculate_label_metrics(i_tp, i_fp, i_fn)
    o_precision, o_recall, o_f1 = calculate_label_metrics(o_tp, o_fp, o_fn)
    
    metrics = {
        'position': {
            'precision': position_correct / position_pred_total if position_pred_total > 0 else 0,
            'recall': position_correct / position_true_total if position_true_total > 0 else 0,
            'f1': 2 * position_correct / (position_pred_total + position_true_total) if (position_pred_total + position_true_total) > 0 else 0,
            'B': {
                'precision': b_precision,
                'recall': b_recall,
                'f1': b_f1,
                'tp': b_tp,
                'fp': b_fp,
                'fn': b_fn
            },
            'I': {
                'precision': i_precision,
                'recall': i_recall,
                'f1': i_f1,
                'tp': i_tp,
                'fp': i_fp,
                'fn': i_fn
            },
            'O': {
                'precision': o_precision,
                'recall': o_recall,
                'f1': o_f1,
                'tp': o_tp,
                'fp': o_fp,
                'fn': o_fn
            }
        },
        'type': {
            'raw_data':{
                'positive_correct': type_positive_correct,
                'positive_total': type_positive_total,
                'positive_top3_correct': type_positive_top3_correct,
                'positive_top5_correct': type_positive_top5_correct,
                'negative_total': type_negative_total,
                'negative_correct': type_negative_correct,
                'mistake_positive_to_negative': type_mistake_positive_to_negative,
                'mistake_negative_to_positive': type_mistake_negative_to_positive
            },
            
            'top1_acc': type_positive_correct / type_positive_total if type_positive_total > 0 else 0,
            'top3_acc': type_positive_top3_correct / type_positive_total if type_positive_total > 0 else 0,
            'top5_acc': type_positive_top5_correct / type_positive_total if type_positive_total > 0 else 0,
            'negative_acc': type_negative_correct / type_negative_total if type_negative_total > 0 else 0,
            'mistake': {
                'positive_to_negative': type_mistake_positive_to_negative/type_positive_total,
                'negative_to_positive': type_mistake_negative_to_positive/type_negative_total
            }
        }
    }
    
    return metrics



    
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, 
                device, num_epochs, save_dir,
                id2type_label=None):
    """
    联合训练模型
    Args:
        model: 模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        num_epochs: 训练轮数
        save_dir: 模型保存目录
        position_weight: 联合训练中位置任务的权重 (0-1之间)
        bi_label_weight: B/I标签相对于O标签的权重
        id2type_label: 类型标签id到名称的映射字典
    """
    # 创建日志文件
    log_file = os.path.join(save_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    best_val_loss = float('inf')
    
    def log_message(message):
        """同时写入文件和打印到控制台"""
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    log_message(f"这是正则化损失后的测试，将positionweight为0.5，bi_label_weight为0.15")
    log_message(f"Starting training with {num_epochs} epochs")
    log_message(f"Training samples: {len(train_dataloader.dataset)}")
    log_message(f"Validation samples: {len(val_dataloader.dataset)}")
        
    # 初始化早停机制相关变量
    best_weighted_score = -1  # 最佳加权得分
    patience = 4  # 最大耐心值
    patience_counter = 0  # 耐心计数器
    early_stop = False  # 是否触发早停
    
    for epoch in range(num_epochs):
        # # 动态调整权重
        # position_weight = 0.4 - (0.4 - 0.15) * (epoch / num_epochs)
        # model.position_weight.data = torch.tensor(position_weight, device=device)
        
        
        # # 可以根据需要动态调整权重
        # if epoch <= 2:
        #     model.position_weight.data = torch.tensor(position_weight, device=device)
        # elif epoch <= 4:
        #     model.position_weight.data = torch.tensor(0.2, device=device)
        # else:
        #     model.position_weight.data = torch.tensor(0.15, device=device)
        log_message(f"Epoch {epoch+1}/{num_epochs}")
        log_message(f"Position Weight (Joint Loss): {model.position_weight:.4f}")
        log_message(f"B/I Label Weight: {model.bi_label_weight.item():.4f}")
        
        # 训练阶段
        model.train()
        total_train_loss = 0
        total_train_position_loss = 0
        total_train_type_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # 准备数据
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
            # 在打印前临时设置
            # print('text:',batch['text'])
            # print("Position labels:")
            # print(batch['position_labels'])
            # print("\nTarget positions:")
            # print(batch['target_positions'])
            # print("\nType labels:")
            # print(batch['type_labels'])
            
            # 计算损失
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_features=dict_features,
                position_labels=position_labels,
                target_positions=target_positions,
                type_labels=type_labels,
                train_mode=True
            )
            
            loss = outputs['loss']
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # 累计损失
            total_train_loss += loss.item()
            total_train_position_loss += outputs['position_loss']
            total_train_type_loss += outputs['type_loss']
            
            # 每100个batch输出一次进度
            if (batch_idx + 1) % 100 == 0:
                log_message(f'Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_dataloader)}:')
                log_message(f'  Position Loss: {outputs["position_loss"]:.4f}')
                log_message(f'  Type Loss: {outputs["type_loss"]:.4f}')
                log_message(f'  Total Loss: {loss.item():.4f}')
                log_message(f'  Position Weight: {model.position_weight:.4f}')
                log_message(f'  B/I Label Weight: {model.bi_label_weight.item():.4f}')
        
        # 计算平均训练损失
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_position_loss = total_train_position_loss / len(train_dataloader)
        avg_train_type_loss = total_train_type_loss / len(train_dataloader)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        total_val_position_loss = 0
        total_val_type_loss = 0
        
        # 收集验证数据用于评估
        all_val_outputs = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                position_labels = batch['position_labels'].to(device)
                target_positions = batch['target_positions'].to(device)
                type_labels = batch['type_labels'].to(device)
                
                dict_features = {k: v.to(device) for k, v in batch['dict_features'].items()}
                
                # 1. 计算损失（使用所有标签）
                loss_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    dict_features=dict_features,
                    position_labels=position_labels,
                    target_positions=target_positions,
                    type_labels=type_labels,
                    train_mode=True #训练模式下才返回损失
                )
                
                # 累计损失
                total_val_loss += loss_outputs['loss'].item()
                total_val_position_loss += loss_outputs['position_loss']
                total_val_type_loss += loss_outputs['type_loss']
                
                # 2. 获取预测
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
                
                cleaned_type_labels = []
                batch_size = type_labels.size(0)
                max_type_len = type_labels.size(1)
                # 清理type_labels
                for batch_idx in range(batch_size):
                    for type_idx in range(max_type_len):
                        if target_positions[batch_idx][type_idx].sum() > 0:  # 跳过填充的位置
                            cleaned_type_labels.append(type_labels[batch_idx][type_idx])
                                
                all_val_outputs.append({
                    'position_predictions': position_pred_outputs['position_predictions'],
                    'type_predictions': type_pred_outputs['type_predictions']
                })
                
                all_val_labels.append({
                    'position_labels': position_labels,
                    'type_labels': cleaned_type_labels, # 清理后的标签 [batch_all_nums]
                    'attention_mask': attention_mask
                })
        
        # 计算平均验证损失
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_position_loss = total_val_position_loss / len(val_dataloader)
        avg_val_type_loss = total_val_type_loss / len(val_dataloader)
        
        # 使用收集的数据计算评估指标
        metrics = evaluate_metrics_from_outputs(all_val_outputs, all_val_labels)
        
        # # 计算加权得分
        # weighted_score = calculate_weighted_score(metrics)
        
        # # 保存最佳模型（基于加权得分）
        # if weighted_score >= best_weighted_score:
        #     best_weighted_score = weighted_score
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': avg_val_loss,
        #         'weighted_score': best_weighted_score,
        #     }, os.path.join(save_dir, 'best_model.pt'))
        #     log_message(f'Saved new best model with weighted score: {best_weighted_score:.4f}')
        #     patience_counter = 0  # 重置耐心计数器
        # else:
        #     patience_counter += 1  # 增加耐心计数器
        
        # 保存最佳模型
        if epoch >= 10:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'weighted_score': best_weighted_score,
            }, os.path.join(save_dir, f'best_model_epoch_{epoch}.pt'))
            log_message(f'Saved new best model with epoch: {epoch:.4f}')

        # 记录每个epoch的训练信息
        log_message(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        log_message(f'Training Loss:')
        log_message(f'  Position Loss: {avg_train_position_loss:.4f}')
        log_message(f'  Type Loss: {avg_train_type_loss:.4f}')
        log_message(f'  Total Loss: {avg_train_loss:.4f}')
        log_message(f'Validation Loss:')
        log_message(f'  Position Loss: {avg_val_position_loss:.4f}')
        log_message(f'  Type Loss: {avg_val_type_loss:.4f}')
        log_message(f'  Total Loss: {avg_val_loss:.4f}')
        
        # 记录评估指标
        log_message(f'\nPerformance Metrics:')
        log_message(f'Position Recognition:')
        log_message(f'  Overall:')
        log_message(f'    Precision: {metrics["position"]["precision"]:.4f}')
        log_message(f'    Recall: {metrics["position"]["recall"]:.4f}')
        log_message(f'    F1: {metrics["position"]["f1"]:.4f}')
        
        # 添加BIO标签的详细信息
        log_message(f'  B Label:')
        log_message(f'    Precision: {metrics["position"]["B"]["precision"]:.4f}')
        log_message(f'    Recall: {metrics["position"]["B"]["recall"]:.4f}')
        log_message(f'    F1: {metrics["position"]["B"]["f1"]:.4f}')
        log_message(f'    TP: {metrics["position"]["B"]["tp"]}, FP: {metrics["position"]["B"]["fp"]}, FN: {metrics["position"]["B"]["fn"]}')
        
        log_message(f'  I Label:')
        log_message(f'    Precision: {metrics["position"]["I"]["precision"]:.4f}')
        log_message(f'    Recall: {metrics["position"]["I"]["recall"]:.4f}')
        log_message(f'    F1: {metrics["position"]["I"]["f1"]:.4f}')
        log_message(f'    TP: {metrics["position"]["I"]["tp"]}, FP: {metrics["position"]["I"]["fp"]}, FN: {metrics["position"]["I"]["fn"]}')
        
        log_message(f'  O Label:')
        log_message(f'    Precision: {metrics["position"]["O"]["precision"]:.4f}')
        log_message(f'    Recall: {metrics["position"]["O"]["recall"]:.4f}')
        log_message(f'    F1: {metrics["position"]["O"]["f1"]:.4f}')
        log_message(f'    TP: {metrics["position"]["O"]["tp"]}, FP: {metrics["position"]["O"]["fp"]}, FN: {metrics["position"]["O"]["fn"]}')
        
        log_message(f'Type Recognition:')
        log_message(f'  Top-1 Accuracy: {metrics["type"]["top1_acc"]:.4f}')
        log_message(f'  Top-3 Accuracy: {metrics["type"]["top3_acc"]:.4f}')
        log_message(f'  Top-5 Accuracy: {metrics["type"]["top5_acc"]:.4f}')
        log_message(f'  Mistake:')
        log_message(f'    Positive to Negative: {metrics["type"]["mistake"]["positive_to_negative"]:.4f}')
        log_message(f'    Negative to Positive: {metrics["type"]["mistake"]["negative_to_positive"]:.4f}')
        log_message(f'raw_data:')
        log_message(f'    Positive Correct: {metrics["type"]["raw_data"]["positive_correct"]}')
        log_message(f'    Positive Total: {metrics["type"]["raw_data"]["positive_total"]}')
        log_message(f'    Positive Top3 Correct: {metrics["type"]["raw_data"]["positive_top3_correct"]}')
        log_message(f'    Positive Top5 Correct: {metrics["type"]["raw_data"]["positive_top5_correct"]}')
        log_message(f'    Negative Total: {metrics["type"]["raw_data"]["negative_total"]}')
        log_message(f'    Negative Correct: {metrics["type"]["raw_data"]["negative_correct"]}')
        log_message('='*50)
        
        # # 检查是否触发早停
        # if patience_counter >= patience:
        #     early_stop = True
        #     log_message(f'Early stopping triggered at epoch {epoch+1}. Best weighted score: {best_weighted_score:.4f}')
        #     break

def get_optimizer_and_scheduler(model, train_dataloader, num_epochs):
    """获取优化器和学习率调度器"""
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if 'bert' in n],
            'lr': 3e-5  # 略微提高BERT的学习率
        },
        {
            'params': [p for n, p in model.named_parameters() if 'bert' not in n],
            'lr': 2e-4  # 提高其他层的学习率
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters)
    
    # 调整预热策略
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * 0.15)  # 增加预热步数到15%
    
    # 使用带有多个循环的余弦退火
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=2.0  # 增加到2个余弦周期
    )
    
    return optimizer, scheduler



    
def main():
    # 基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 添加设备信息打印
    print(f"\n=== 训练设备信息 ===")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"当前GPU显存使用: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"当前GPU显存缓存: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    print("="*30 + "\n")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 预处理特征和映射文件路径
    features_path = os.path.join(DATA_DIR, 'allusion_features_strictly_dict.pt')
    mapping_path = os.path.join(DATA_DIR, 'allusion_mapping_strictly_dict.json')
    
    # 检查预处理文件是否存在
    if not os.path.exists(features_path) or not os.path.exists(mapping_path):
        raise FileNotFoundError("请先运行 process_dict_features.py 生成预处理特征！")
    
    # 加载典故词典和类型映射
    allusion_dict, type_label2id, id2type_label, num_types = load_allusion_dict()
    dict_size = len(allusion_dict)
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    # 初始化模型
    model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size,
                            bi_label_weight=0.15,position_weight=0.65).to(device)

    print("\nstarting from scratch")
    # 创建训练和验证数据集
    train_dataset = PoetryNERDataset(
        os.path.join(DATA_DIR, '4_train_position_no_bug_less_negatives.csv'),
        tokenizer, MAX_SEQ_LEN,
        type_label2id=type_label2id,
        id2type_label=id2type_label,
        features_path=features_path,
        mapping_path=mapping_path,
        negative_sample_ratio=0.05
    )
    
    val_dataset = PoetryNERDataset(
        os.path.join(DATA_DIR, '4_val_position_no_bug_less_negatives.csv'),
        tokenizer, MAX_SEQ_LEN,
        type_label2id=type_label2id,
        id2type_label=id2type_label,
        features_path=features_path,
        mapping_path=mapping_path,
        negative_sample_ratio=0.05
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 验证集不需要打乱
        collate_fn=val_dataset.collate_fn
    )
    
    # 获取优化器和调度器
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        train_dataloader=train_dataloader,
        num_epochs=EPOCHS
    )
    
    # 开始训练
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=EPOCHS,
        save_dir=SAVE_DIR,
        id2type_label=id2type_label
    )

if __name__ == '__main__':
    main()
