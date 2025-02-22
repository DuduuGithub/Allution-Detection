import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from poetry_dataset import PoetryNERDataset
from bert_crf import AllusionBERTCRF
import os
from config import MODEL_NAME, BERT_MODEL_PATH, MAX_SEQ_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAIN_PATH, TEST_PATH, SAVE_DIR, ALLUSION_TYPES_PATH
import argparse

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs, save_dir, task):
    best_val_loss = float('inf')
    
    # 在函数开始时创建文件
    train_log_file = os.path.join(save_dir, f'train_loss_{task}.txt')
    val_log_file = os.path.join(save_dir, f'val_loss_{task}.txt')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_labels = batch['position_labels'].to(device)
            
            optimizer.zero_grad()
            
            if task == 'position':
                loss = model(input_ids, attention_mask, labels=position_labels)
            else:  # task == 'type'
                type_labels = batch['type_labels'].to(device)
                loss = model(input_ids, attention_mask, labels=position_labels, type_labels=type_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}')
        
        avg_train_loss = total_loss / len(train_dataloader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                position_labels = batch['position_labels'].to(device)
                
                if task == 'position':
                    loss = model(input_ids, attention_mask, labels=position_labels)
                else:  # task == 'type'
                    type_labels = batch['type_labels'].to(device)
                    loss = model(input_ids, attention_mask, labels=position_labels, type_labels=type_labels)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        # 保存训练和验证损失到文件
        with open(train_log_file, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch+1}: {avg_train_loss:.4f}\n')
        
        with open(val_log_file, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch+1}: {avg_val_loss:.4f}\n')
        
        # 同时打印到控制台
        print(f'Epoch {epoch+1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(save_dir, f'best_model_{task}.pt'))

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, choices=['position', 'type'], required=True,
                      help='Training stage: position (Stage 1) or type (Stage 2)')
    args = parser.parse_args()
    
    # 基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = MODEL_NAME
    max_len = MAX_SEQ_LEN
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    learning_rate = LEARNING_RATE
    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    # 获取典故类型数量
    type_label2id, _ = load_allusion_types(ALLUSION_TYPES_PATH)
    num_types = len(type_label2id)
    print(f"Total number of allusion types: {num_types}")
    
    if args.stage == 'position':
        # 阶段一：典故位置识别
        print("开始训练阶段一：典故位置识别")
        model = AllusionBERTCRF(num_types=num_types, task='position').to(device)
        
        # 加载位置识别数据集
        train_dataset = PoetryNERDataset(TRAIN_PATH, tokenizer, max_len, task='position')
        val_dataset = PoetryNERDataset(TEST_PATH, tokenizer, max_len, task='position')
        
    else:  # args.stage == 'type'
        # 阶段二：典故类型分类
        print("开始训练阶段二：典故类型分类")
        model = AllusionBERTCRF(num_types=num_types, task='type').to(device)
        
        # 检查是否存在阶段一的模型
        position_model_path = os.path.join(save_dir, 'best_model_position.pt')
        if not os.path.exists(position_model_path):
            raise FileNotFoundError("请先完成阶段一（position）的训练！")
            
        # 加载位置识别模型的参数
        checkpoint = torch.load(position_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载类型分类数据集
        train_dataset = PoetryNERDataset(TRAIN_PATH, tokenizer, max_len, task='type')
        val_dataset = PoetryNERDataset(TEST_PATH, tokenizer, max_len, task='type')
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps * 0.1,
        num_training_steps=total_steps
    )
    
    # 训练模型
    train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, 
                device, epochs, save_dir, args.stage)

if __name__ == '__main__':
    main()
