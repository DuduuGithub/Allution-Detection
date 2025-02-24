import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from poetry_dataset import PoetryNERDataset
from bert_crf import AllusionBERTCRF, prepare_sparse_features
import os
from config import (
    MODEL_NAME, BERT_MODEL_PATH, MAX_SEQ_LEN, BATCH_SIZE, 
    EPOCHS, LEARNING_RATE, TRAIN_PATH, TEST_PATH, 
    SAVE_DIR, ALLUSION_TYPES_PATH
)
import argparse
import pandas as pd

def load_allusion_dict(dict_file='data/典故的代表词组.csv'):
    """加载典故词典并创建类型映射
    
    Returns:
        tuple: (
            allusion_dict: Dict[str, List[str]],  # {典故代表词: [变体列表]}
            type_label2id: Dict[str, int],        # {类型名: 类型ID}
            id2type_label: Dict[int, str],        # {类型ID: 类型名}
            num_types: int                        # 类型总数
        )
    """
    # 1. 读取典故词典
    allusion_dict = {}
    type_set = set()  # 用于收集所有类型
    
    df = pd.read_csv(dict_file, encoding='utf-8')
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
    # 对类型名称排序以确保映射的一致性
    sorted_types = sorted(list(type_set))
    type_label2id = {label: idx for idx, label in enumerate(sorted_types)}
    id2type_label = {idx: label for label, idx in type_label2id.items()}
    num_types = len(type_label2id)
    
    print(f"Loaded allusion dictionary with {len(allusion_dict)} entries")
    print(f"Found {num_types} unique allusion types")
    
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
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        # 保存训练和验证损失
        with open(train_log_file, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch+1}: {avg_train_loss:.4f}\n')
        
        with open(val_log_file, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch+1}: {avg_val_loss:.4f}\n')
        
        print(f'Epoch {epoch+1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        
        # 使用统一的模型保存文件名
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'task': task,  # 保存当前任务信息
            }, os.path.join(save_dir, 'best_model.pt'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, choices=['position', 'type'], 
                       required=True, help='Training stage: position or type')
    args = parser.parse_args()
    
    # 基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 加载典故词典和类型映射
    allusion_dict, type_label2id, id2type_label, num_types = load_allusion_dict()
    dict_size = len(allusion_dict)
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    if args.stage == 'position':
        print("Starting Stage 1: Position Recognition")
        model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size).to(device)
        
        train_dataset = PoetryNERDataset(
            TRAIN_PATH, tokenizer, MAX_SEQ_LEN,
            type_label2id=type_label2id,
            id2type_label=id2type_label,
            task='position'
        )
        val_dataset = PoetryNERDataset(
            TEST_PATH, tokenizer, MAX_SEQ_LEN,
            type_label2id=type_label2id,
            id2type_label=id2type_label,
            task='position'
        )
        
    else:  # args.stage == 'type'
        print("Starting Stage 2: Type Classification")
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
            TRAIN_PATH, tokenizer, MAX_SEQ_LEN,
            type_label2id=type_label2id,
            id2type_label=id2type_label,
            task='type'
        )
        val_dataset = PoetryNERDataset(
            TEST_PATH, tokenizer, MAX_SEQ_LEN,
            type_label2id=type_label2id,
            id2type_label=id2type_label,
            task='type'
        )
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
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

if __name__ == '__main__':
    main()
