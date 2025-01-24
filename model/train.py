import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from poetry_dataset import PoetryNERDataset
from bert_crf import AllusionBERTCRF
import os
from config import MODEL_NAME, BERT_MODEL_PATH, MAX_SEQ_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAIN_PATH, VAL_PATH, SAVE_DIR, ALLUSION_TYPES_PATH

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs, save_dir, task):
    best_val_loss = float('inf')
    
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
                position_labels = batch['position_labels'].to(device)
                
                if task == 'position':
                    loss = model(input_ids, attention_mask, labels=position_labels)
                else:  # task == 'type'
                    type_labels = batch['type_labels'].to(device)
                    loss = model(input_ids, attention_mask, labels=position_labels, type_labels=type_labels)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        
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
    # 基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = MODEL_NAME
    max_len = MAX_SEQ_LEN
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    learning_rate = LEARNING_RATE
    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取guwenbert-large的绝对路径
    model_path = os.path.join(os.path.dirname(current_dir), 'model', model_name)
    
    print(f"尝试加载模型，路径: {model_path}")
        
    # 首先初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # 获取典故类型数量
    train_dataset = PoetryNERDataset(TRAIN_PATH, tokenizer, max_len, task='type')
    num_types = len(train_dataset.type_label2id)
    
    # 阶段一：典故位置识别
    print("开始训练阶段一：典故位置识别")
    model = AllusionBERTCRF(num_types=num_types, task='position').to(device)
    
    # 加载位置识别数据集
    train_dataset = PoetryNERDataset(TRAIN_PATH, tokenizer, max_len, task='position')
    val_dataset = PoetryNERDataset(VAL_PATH, tokenizer, max_len, task='position')
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps * 0.1,
        num_training_steps=total_steps
    )
    
    train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, 
                device, epochs, save_dir, 'position')
    
    # 阶段二：典故类型分类
    print("\n开始训练阶段二：典故类型分类")
    # 加载位置识别模型的参数
    checkpoint = torch.load(os.path.join(save_dir, 'best_model_position.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.task = 'type'  # 切换到类型分类任务
    
    # 加载类型分类数据集
    train_dataset = PoetryNERDataset(TRAIN_PATH, tokenizer, max_len, task='type')
    val_dataset = PoetryNERDataset(VAL_PATH, tokenizer, max_len, task='type')
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps * 0.1,
        num_training_steps=total_steps
    )
    
    train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, 
                device, epochs, save_dir, 'type')

if __name__ == '__main__':
    main()
