import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.poetry_dataset import PoetryNERDataset
from model.bert_crf import AllusionBERTCRF, prepare_sparse_features
from model.config import (
    BERT_MODEL_PATH, MAX_SEQ_LEN, BATCH_SIZE, 
    POSITION_EPOCHS, TYPE_EPOCHS, LEARNING_RATE,
    SAVE_DIR, DATA_DIR, ALLUSION_DICT_PATH
)
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import argparse
import pandas as pd


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

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, 
                device, num_epochs, save_dir, task, id2type_label=None):
    """
    训练模型
    Args:
        model: 模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        num_epochs: 训练轮数
        save_dir: 模型保存目录
        task: 任务类型 ('position' 或 'type')
        id2type_label: 类型标签id到名称的映射字典
    """
    if task == 'type' and id2type_label is None:
        raise ValueError("id2type_label must be provided for type classification task")

    # 添加早停相关变量
    patience = 3  # 连续3次无改善则停止
    min_improvement = 0.005  # 最小改善阈值0.5%
    no_improvement_count = 0
    best_top1_accuracy = 0
    
    best_val_loss = float('inf')
    best_f1 = 0
    best_accuracy = 0
    
    # 计算打印频率
    total_batches = len(train_dataloader)
    print_freq = 50
    
    # 日志文件
    train_log_file = os.path.join(save_dir, f'train_loss_{task}.txt')
    val_log_file = os.path.join(save_dir, f'val_loss_{task}.txt')
    batch_log_file = os.path.join(save_dir, f'batch_loss_{task}.txt')
    
    # 在训练开始时记录超参数
    print(f"\nTraining {task.capitalize()} Task with parameters:")
    if task == 'position':
        print(f"B/I Label Weight: {model.pos_weight}")
        with open(train_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n=== Training with Parameters ===\n")
            f.write(f"B/I Label Weight: {model.pos_weight}\n")
        with open(val_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n=== Training with Parameters ===\n")
            f.write(f"B/I Label Weight: {model.pos_weight}\n")
    
    print(f"Total batches per epoch: {total_batches}")
    print(f"Will print progress every {print_freq} batches")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        batch_losses = []
        
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 将字典中的每个张量移到device
            dict_features = {
                'indices': batch['dict_features']['indices'].to(device),
                'values': batch['dict_features']['values'].to(device),
                'active_counts': batch['dict_features']['active_counts'].to(device)
            }
            
            optimizer.zero_grad()
            
            if task == 'position':
                position_labels = batch['position_labels'].to(device)

                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    dict_features=dict_features,
                    task='position',
                    position_labels=position_labels
                )
            else:  # task == 'type'
                
                type_labels = batch['type_labels'].to(device)
                target_positions = batch['target_positions'].to(device)
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    dict_features=dict_features,
                    task='type',
                    target_positions=target_positions,
                    type_labels=type_labels
                )
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 优化器步进
            optimizer.step()
            scheduler.step()  # 更新学习率
            optimizer.zero_grad()
            
            # 记录当前学习率
            current_lr = scheduler.get_last_lr()[0]
            
            # 记录当前batch的损失
            current_loss = loss.item()
            batch_losses.append(current_loss)
            total_loss += current_loss
            
            # 每print_freq个batch打印一次平均损失
            if (batch_idx + 1) % print_freq == 0:
                # 计算最近print_freq个batch的平均损失
                recent_avg_loss = sum(batch_losses[-print_freq:]) / print_freq
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}/{total_batches}, '
                      f'Recent Average Loss: {recent_avg_loss:.4f}, '
                      f'Current Batch Loss: {current_loss:.4f}')
        
        # 计算整个epoch的平均损失
        epoch_avg_loss = total_loss / total_batches
        
        # 记录每个epoch的训练损失
        with open(train_log_file, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch+1}: {epoch_avg_loss:.4f}\n')
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_steps = 0
        
        if task == 'position':
            # 位置识别任务的统计变量
            b_tp, b_fp, b_fn = 0, 0, 0
            i_tp, i_fp, i_fn = 0, 0, 0
            o_tp, o_fp, o_fn = 0, 0, 0
        else:  # task == 'type'
            # 类型识别任务的统计变量
            val_predictions = []
            val_labels = []
            top1_correct = 0
            top3_correct = 0
            top5_correct = 0
            total = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # 将字典中的每个张量移到device
                dict_features = {
                    'indices': batch['dict_features']['indices'].to(device),
                    'values': batch['dict_features']['values'].to(device),
                    'active_counts': batch['dict_features']['active_counts'].to(device)
                }
                
                if task == 'position':
                    position_labels = batch['position_labels'].to(device)
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
                        for p, l in zip(pred, label):
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
                                
                if task == 'type':
                    type_labels = batch['type_labels'].to(device)
                    target_positions = batch['target_positions'].to(device)
                    
                    
                    # 获取损失值
                    loss = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        dict_features=dict_features,
                        task='type',
                        target_positions=target_positions,
                        type_labels=type_labels
                    )
                    
                    # 获取预测结果
                    pred_top5 = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        dict_features=dict_features,
                        task='type',
                        target_positions=target_positions
                    )
                    
                    predictions = pred_top5['predictions']  # [batch_size, 5]
                    probabilities = pred_top5['probabilities']  # [batch_size, 5]
                    
                    # 计算top-k准确率
                    batch_size = type_labels.size(0)
                    total += batch_size
                    for pred_top_k, label in zip(predictions, type_labels):
                        if label in pred_top_k[:1]:  # top1
                            top1_correct += 1
                        if label in pred_top_k[:3]:  # top3
                            top3_correct += 1
                        if label in pred_top_k[:5]:  # top5
                            top5_correct += 1
                    
                    # 收集预测和真实标签（只收集top1预测用于混淆矩阵）
                    val_predictions.extend(predictions[:, 0].cpu().tolist())
                    val_labels.extend(type_labels.cpu().tolist())
                    
                val_loss += loss.item()
                val_steps += 1
        
        # 计算整个验证集的统计信息
        val_loss = val_loss / val_steps
        
        # 计算并打印详细指标
        print("\nValidation Results:")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # 计算并打印每个标签的详细指标
        def calculate_metrics(tp, fp, fn):
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            return precision, recall, f1
        
        if task == 'position':
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
            
            # 使用macro_f1作为保存指标
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_loss': val_loss,
                    'macro_f1': macro_f1,  # 保存F1指标
                    'task': task,
                }, os.path.join(save_dir, f'best_model_{task}.pt'))
                print(f"\nModel saved with new best macro F1: {macro_f1:.4f}")
            else:
                print(f"\nNo improvement in macro F1. Current best: {best_f1:.4f}")
            
            # 保存训练和验证损失
            with open(val_log_file, 'a', encoding='utf-8') as f:
                f.write(f'Epoch {epoch+1}: {val_loss:.4f}\n')
                f.write(f"Validation Loss: {val_loss:.4f}\n")
                f.write(f"B Label - Precision: {b_precision:.4f}, Recall: {b_recall:.4f}, F1: {b_f1:.4f}\n")
                f.write(f"         TP: {b_tp}, FP: {b_fp}, FN: {b_fn}\n")
                f.write(f"I Label - Precision: {i_precision:.4f}, Recall: {i_recall:.4f}, F1: {i_f1:.4f}\n")
                f.write(f"         TP: {i_tp}, FP: {i_fp}, FN: {i_fn}\n")
                f.write(f"O Label - Precision: {o_precision:.4f}, Recall: {o_recall:.4f}, F1: {o_f1:.4f}\n")
                f.write(f"         TP: {o_tp}, FP: {o_fp}, FN: {o_fn}\n")
                f.write(f"\nMacro Average - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}\n")
        
        elif task == 'type':
            print("\nType Task Detailed Metrics:")
            # 计算准确率
            top1_accuracy = top1_correct / total
            top3_accuracy = top3_correct / total
            top5_accuracy = top5_correct / total
            
            
            # 打印验证结果
            print(f"\n=== Type Classification Results ===")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
            print(f"Top-3 Accuracy: {top3_accuracy:.4f}")
            print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
            
            print("\nTop 10 Most Confused Type Pairs:")
            for (true_type, pred_type), count in confusion_pairs[:10]:
                print(f"True: {true_type} -> Predicted: {pred_type}, Count: {count}")
            
            # 记录到日志文件
            with open(val_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\nEpoch {epoch+1}:\n")
                f.write(f"Validation Loss: {val_loss:.4f}\n")
                f.write(f"Top-1 Accuracy: {top1_accuracy:.4f}\n")
                f.write(f"Top-3 Accuracy: {top3_accuracy:.4f}\n")
                f.write(f"Top-5 Accuracy: {top5_accuracy:.4f}\n")
                f.write("\nTop 10 Most Confused Type Pairs:\n")
                for (true_type, pred_type), count in confusion_pairs[:10]:
                    f.write(f"True: {true_type} -> Predicted: {pred_type}, Count: {count}\n")
            
            # 使用top1准确率作为模型保存的指标
            if top1_accuracy > best_top1_accuracy:
                best_top1_accuracy = top1_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_loss': val_loss,
                    'top1_accuracy': top1_accuracy,
                    'top3_accuracy': top3_accuracy,
                    'top5_accuracy': top5_accuracy,
                }, os.path.join(save_dir, 'best_model_type.pt'))
                print(f"\nModel saved with new best top-1 accuracy: {top1_accuracy:.4f}")
            
            # 早停检查
            if top1_accuracy < best_top1_accuracy - min_improvement:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"\nNo improvement in top-1 accuracy for {patience} epochs. Stopping training.")
                    break
        
        # 原有的基于损失的保存逻辑可以保留，但改为保存一个单独的文件
        

def get_optimizer_and_scheduler(model, train_dataloader, num_epochs, current_epoch,
                              task='position', warmup_ratio=0.1, initial_lr=2e-5):
    """获取优化器和学习率调度器"""
    # 获取共享参数的学习率
    shared_lr = model.adjust_shared_learning_rate(current_epoch, initial_lr)
    
    # 创建参数组，确保每个参数只出现在一个组中
    shared_params = []
    position_params = []
    type_params = []
    
    for name, param in model.named_parameters():
        # 首先检查任务特定参数
        if any(layer in name for layer in ['position_classifier', 'position_crf']):
            position_params.append(param)
        elif any(layer in name for layer in ['type_classifier', 'attention']):
            type_params.append(param)
        # 其他都是共享参数
        else:
            shared_params.append(param)
    
    optimizer_grouped_parameters = [
        # 共享参数组
        {
            'params': shared_params,
            'lr': shared_lr
        },
        # 位置识别特定参数
        {
            'params': position_params,
            'lr': initial_lr * (5 if task == 'position' else 0.1)
        },
        # 类型识别特定参数
        {
            'params': type_params,
            'lr': initial_lr * (5 if task == 'type' else 0.1)
        }
    ]
    
    # 只添加非空的参数组
    final_param_groups = [
        group for group in optimizer_grouped_parameters
        if len(group['params']) > 0
    ]
    
    # 创建优化器
    optimizer = AdamW(final_param_groups)
    
    # 计算总训练步数
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5
    )
    
    # 打印参数组信息以便调试
    print("\nOptimizer parameter groups:")
    for i, group in enumerate(final_param_groups):
        param_count = sum(p.numel() for p in group['params'])
        print(f"Group {i}: {param_count} parameters, learning rate: {group['lr']}")
    
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
    model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size).to(device)
    
    # 检查是否有已存在的模型文件
    position_checkpoint_path = os.path.join(SAVE_DIR, 'best_model_position.pt')
    type_checkpoint_path = os.path.join(SAVE_DIR, 'best_model_type.pt')
    
    # 如果有已存在的模型文件，加载最新的参数
    if os.path.exists(position_checkpoint_path):
        print("\nFound existing position model checkpoint")
        checkpoint = torch.load(position_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded position model from epoch {checkpoint['epoch']}")
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("\nNo existing model checkpoint found, starting from scratch")
        start_epoch = 0
    
    # 创建两种任务的训练和验证数据集
    position_train_dataset = PoetryNERDataset(
        os.path.join(DATA_DIR, '4_train_position_no_bug.csv'),
        tokenizer, MAX_SEQ_LEN,
        type_label2id=type_label2id,
        id2type_label=id2type_label,
        task='position',
        features_path=features_path,
        mapping_path=mapping_path
    )
    
    position_val_dataset = PoetryNERDataset(
        os.path.join(DATA_DIR, '4_val_position_no_bug.csv'),
        tokenizer, MAX_SEQ_LEN,
        type_label2id=type_label2id,
        id2type_label=id2type_label,
        task='position',
        features_path=features_path,
        mapping_path=mapping_path
    )
    
    type_train_dataset = PoetryNERDataset(
        os.path.join(DATA_DIR, '4_train_type_no_bug.csv'),
        tokenizer, MAX_SEQ_LEN,
        type_label2id=type_label2id,
        id2type_label=id2type_label,
        task='type',
        features_path=features_path,
        mapping_path=mapping_path
    )
    
    type_val_dataset = PoetryNERDataset(
        os.path.join(DATA_DIR, '4_val_type_no_bug.csv'),
        tokenizer, MAX_SEQ_LEN,
        type_label2id=type_label2id,
        id2type_label=id2type_label,
        task='type',
        features_path=features_path,
        mapping_path=mapping_path
    )
    
    # 创建数据加载器
    position_train_dataloader = DataLoader(
        position_train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=position_train_dataset.collate_fn
    )
    
    position_val_dataloader = DataLoader(
        position_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 验证集不需要打乱
        collate_fn=position_val_dataset.collate_fn
    )
    
    type_train_dataloader = DataLoader(
        type_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=type_train_dataset.collate_fn
    )
    
    type_val_dataloader = DataLoader(
        type_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 验证集不需要打乱
        collate_fn=type_val_dataset.collate_fn
    )
    
    # 交替训练
    total_epochs = 30  # 总训练轮数
    epochs_per_switch = 2  # 每次切换任务前训练的轮数
    
    for epoch in range(start_epoch, total_epochs):
        current_task = 'position' if (epoch // epochs_per_switch) % 2 == 0 else 'type'
        current_train_dataloader = position_train_dataloader if current_task == 'position' else type_train_dataloader
        current_val_dataloader = position_val_dataloader if current_task == 'position' else type_val_dataloader
        
        # 获取针对当前任务的优化器和调度器
        optimizer, scheduler = get_optimizer_and_scheduler(
            model, current_train_dataloader, epochs_per_switch, 
            current_epoch=epoch, task=current_task
        )
        
        print(f"\nEpoch {epoch+1}/{total_epochs}, Training {current_task} task")
        
        # 训练当前任务
        train_model(
            model=model,
            train_dataloader=current_train_dataloader,
            val_dataloader=current_val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=1,
            save_dir=SAVE_DIR,
            task=current_task,
            id2type_label=id2type_label
        )
        
        # 每个epoch结束后保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'task': current_task,
        }
        torch.save(checkpoint, os.path.join(SAVE_DIR, f'latest_model_{current_task}.pt'))

def test():
    """测试训练相关功能"""
    # 测试用例
    test_cases = [
        ("一入石渠署，三闻宫树蝉。", [(2, 3, '石渠')]),
        ("桃源避秦人不见，武陵渔父独知处。", [(0, 2, "桃源")]),
    ]
    
    print("=== 测试字典特征提取 ===")
    
    # 加载典故词典
    allusion_dict, _, _, _ = load_allusion_dict()
    
    # 为每个典故分配ID
    allusion_to_id = {name: idx for idx, name in enumerate(allusion_dict.keys())}
    id_to_allusion = {idx: name for name, idx in allusion_to_id.items()}
    
    for poem, expected in test_cases:
        print(f"\n测试诗句: {poem}")
        print(f"预期典故: {expected}")
        
        # 获取特征
        features = prepare_sparse_features([poem], allusion_dict)
        
        print("\n提取的特征:")
        print("特征形状:", features['indices'].shape)
        print("活跃特征数:", features['active_counts'][0])
        
        # 分析每个位置检测到的典故
        print("\n位置分析:")
        for pos in range(len(poem)):
            active_count = features['active_counts'][0][pos].item()
            if active_count > 0:
                print(f"\n位置 {pos} ({poem[pos]}):")
                for idx in range(active_count):
                    allusion_id = features['indices'][0][pos][idx].item()
                    similarity = features['values'][0][pos][idx].item()
                    allusion_name = id_to_allusion[allusion_id]
                    print(f"  - 典故: {allusion_name}, 相似度: {similarity:.3f}")
        
        print("\n" + "="*50)

if __name__ == '__main__':
    main()
