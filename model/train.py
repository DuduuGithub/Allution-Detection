import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from poetry_dataset import PoetryNERDataset
from bert_crf import AllusionBERTCRF, prepare_sparse_features
import os
from config import (
    BERT_MODEL_PATH, MAX_SEQ_LEN, BATCH_SIZE, 
    POSITION_EPOCHS, TYPE_EPOCHS, LEARNING_RATE,
    SAVE_DIR, DATA_DIR, ALLUSION_DICT_PATH
)
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
        variants = eval(row['representatives'])  # 安全地解析字符串列表
        
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
                device, num_epochs, save_dir, task, allusion_dict):
    best_val_loss = float('inf')
    
    # 日志文件
    train_log_file = os.path.join(save_dir, f'train_loss_{task}.txt')
    val_log_file = os.path.join(save_dir, f'val_loss_{task}.txt')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 生成字典特征
            texts = batch['text']  
            dict_features = prepare_sparse_features(texts, allusion_dict)
            dict_features = {
                k: v.to(device) for k, v in dict_features.items()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 1 == 0:
                
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}')
        
        avg_train_loss = total_loss / len(train_dataloader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # 生成字典特征
                texts = batch['text']
                dict_features = prepare_sparse_features(texts, allusion_dict)
                dict_features = {
                    k: v.to(device) for k, v in dict_features.items()
                }
                
                if task == 'position':
                    position_labels = batch['position_labels'].to(device)
                    # 获取损失和预测结果
                    loss, predictions = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        dict_features=dict_features,
                        task='position',
                        position_labels=position_labels,
                        return_predictions=True
                    )
                    # 计算准确率（忽略填充标记）
                    mask = attention_mask.bool()
                    correct = ((predictions == position_labels) & mask).sum().item()
                    total = mask.sum().item()
                    
                else:  # task == 'type'
                    type_labels = batch['type_labels'].to(device)
                    target_positions = batch['target_positions'].to(device)
                    # 获取损失和预测结果
                    loss, predictions = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        dict_features=dict_features,
                        task='type',
                        target_positions=target_positions,
                        type_labels=type_labels,
                        return_predictions=True
                    )
                    # 计算准确率
                    correct = (predictions == type_labels).sum().item()
                    total = len(type_labels)
                
                val_loss += loss.item()
                total_correct += correct
                total_samples += total
        
        avg_val_loss = val_loss / len(val_dataloader)
        accuracy = total_correct / total_samples * 100
        
        # 保存训练和验证损失
        with open(train_log_file, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch+1}: {avg_train_loss:.4f}\n')
        
        with open(val_log_file, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch+1}: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
        
        print(f'Epoch {epoch+1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        print(f'Validation accuracy: {accuracy:.2f}%')
        
        # 使用统一的模型保存文件名
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'accuracy': accuracy,
                'task': task,
            }, os.path.join(save_dir, 'best_model.pt'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, choices=['position', 'type'], 
                       required=True, help='Training stage: position or type')
    args = parser.parse_args()
    
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
    
    # 根据任务选择训练轮数和数据路径
    EPOCHS = POSITION_EPOCHS if args.stage == 'position' else TYPE_EPOCHS
    train_file = os.path.join(DATA_DIR, f'4_train_{args.stage}.csv')
    val_file = os.path.join(DATA_DIR, f'4_val_{args.stage}.csv')
    test_file = os.path.join(DATA_DIR, f'4_test_{args.stage}.csv')
    
    # 加载典故词典和类型映射
    allusion_dict, type_label2id, id2type_label, num_types = load_allusion_dict()
    dict_size = len(allusion_dict)
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    if args.stage == 'position':
        print("Starting Stage 1: Position Recognition")
        print(f"Training for {EPOCHS} epochs")
        print(f"Using training data: {train_file}")
        print(f"Using validation data: {val_file}")
        
        model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size).to(device)
        
        train_dataset = PoetryNERDataset(
            train_file, tokenizer, MAX_SEQ_LEN,
            type_label2id=type_label2id,
            id2type_label=id2type_label,
            task='position'
        )
        val_dataset = PoetryNERDataset(
            val_file, tokenizer, MAX_SEQ_LEN,
            type_label2id=type_label2id,
            id2type_label=id2type_label,
            task='position'
        )
        
    else:  # args.stage == 'type'
        print("Starting Stage 2: Type Classification")
        print(f"Training for {EPOCHS} epochs")
        print(f"Using training data: {train_file}")
        print(f"Using validation data: {val_file}")
        
        model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size).to(device)
        
        # 加载模型参数
        checkpoint_path = os.path.join(SAVE_DIR, 'best_model.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Please complete Stage 1 (position) training first!")
            
        checkpoint = torch.load(checkpoint_path)
        stage1_state_dict = checkpoint['model_state_dict']
        
        # 只加载bert和bilstm的参数
        new_state_dict = {
            name: param for name, param in stage1_state_dict.items()
            if name.startswith('bert') or name.startswith('bilstm')
        }
        model.load_state_dict(new_state_dict, strict=False)
        print("Loaded BERT and BiLSTM parameters from Stage 1 model")
        
        train_dataset = PoetryNERDataset(
            train_file, tokenizer, MAX_SEQ_LEN,
            type_label2id=type_label2id,
            id2type_label=id2type_label,
            task='type'
        )
        val_dataset = PoetryNERDataset(
            val_file, tokenizer, MAX_SEQ_LEN,
            type_label2id=type_label2id,
            id2type_label=id2type_label,
            task='type'
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
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps * 0.1,
        num_training_steps=total_steps
    )
    
    # 训练模型
    train_model(
        model, train_dataloader, val_dataloader, optimizer, scheduler,
        device, EPOCHS, SAVE_DIR, args.stage, allusion_dict
    )

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
