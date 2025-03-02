import sys
import os

# 添加当前目录的父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model.poetry_dataset import PoetryNERDataset
from model.train import load_allusion_dict
from model.bert_crf import AllusionBERTCRF
from model.config import BERT_MODEL_PATH, MAX_SEQ_LEN, SAVE_DIR, TEST_PATH, DATA_DIR, BATCH_SIZE
import heapq
from collections import defaultdict

def analyze_mistakes(model, dataloader, device):
    model.eval()
    mistakes = {
        'B_as_O': [], 'B_as_I': [],
        'I_as_O': [], 'I_as_B': [],
        'O_as_B': [], 'O_as_I': []
    }
    high_loss_cases = []  # 存储高损失案例
    allusion_confusion = defaultdict(lambda: defaultdict(int))  # 典故混淆统计
    
    with torch.no_grad():
        batch_idx = 0  # 添加batch计数器
        for batch in dataloader:
            batch_idx += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 将字典中的每个张量移到device
            dict_features = {
                'indices': batch['dict_features']['indices'].to(device),
                'values': batch['dict_features']['values'].to(device),
                'active_counts': batch['dict_features']['active_counts'].to(device)
            }
            
            position_labels = batch['position_labels'].to(device)
            texts = batch['text']
            
            # 获取每个样本的损失
            losses = []
            for i in range(len(texts)):
                # 计算单个样本的损失
                sample_loss = model(
                    input_ids=input_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1],
                    dict_features={
                        'indices': dict_features['indices'][i:i+1],
                        'values': dict_features['values'][i:i+1],
                        'active_counts': dict_features['active_counts'][i:i+1]
                    },
                    task='position',
                    position_labels=position_labels[i:i+1],
                )
                losses.append(sample_loss.item())
            
            # 获取预测结果
            predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_features=dict_features,
                task='position'
            )
            
            # 计算每个样本的损失
            for idx, (text, pred, label, mask, sample_loss) in enumerate(
                zip(texts, predictions, position_labels[:, 1:-1], 
                    attention_mask[:, 1:-1], losses)):
                
                # 提取典故片段
                true_allusions = extract_allusion(text, label)
                pred_allusions = extract_allusion(text, pred)
                
                # 将列表转换为字符串，用于字典键
                true_allusion_str = '|'.join(true_allusions) if true_allusions else 'NONE'
                pred_allusion_str = '|'.join(pred_allusions) if pred_allusions else 'NONE'
                
                # 如果预测错误，更新典故混淆统计
                if true_allusion_str != pred_allusion_str:
                    allusion_confusion[true_allusion_str][pred_allusion_str] += 1
                
                # 创建案例字典
                case = {
                    'text': text,
                    'loss': sample_loss,
                    'true_labels': label.tolist(),
                    'pred_labels': pred,
                    'true_allusion': true_allusions,
                    'pred_allusion': pred_allusions
                }
                
                # 保存高损失案例
                heapq.heappush(high_loss_cases, (-sample_loss, batch_idx, idx, case))
                
                # 分析错误类型
                for pos, (p, l, m) in enumerate(zip(pred, label, mask)):
                    if not m:  # 跳过padding位置
                        continue
                    
                    if l != p:  # 预测错误
                        mistake_type = f"{['O','B','I'][l]}_as_{['O','B','I'][p]}"
                        if mistake_type in mistakes:
                            context = text[max(0, pos-10):min(len(text), pos+10)]
                            mistakes[mistake_type].append({
                                'text': text,
                                'position': pos,
                                'context': context,
                                'char': text[pos],
                                'true_labels': label.tolist(),
                                'pred_labels': pred
                            })
    
    return mistakes, high_loss_cases, allusion_confusion

def extract_allusion(text, labels):
    """提取文本中的典故片段"""
    allusions = []
    current = []
    for i, label in enumerate(labels):
        if label == 1:  # B
            if current:
                allusions.append(''.join(current))
            current = [text[i]]
        elif label == 2:  # I
            if current:
                current.append(text[i])
        elif label == 0:  # O
            if current:
                allusions.append(''.join(current))
                current = []
    if current:
        allusions.append(''.join(current))
    return allusions

def format_labels(text, labels):
    """格式化标签展示"""
    result = []
    for char, label in zip(text, labels):
        if label == 1:
            result.append(f"[B]{char}")
        elif label == 2:
            result.append(f"[I]{char}")
        else:
            result.append(char)
    return ''.join(result)

def main():
    # 加载模型和数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)    
    # 加载checkpoint以获取正确的参数大小
    checkpoint = torch.load(os.path.join(SAVE_DIR, 'best_model.pt'))
    
    # 加载典故词典和类型映射
    allusion_dict, type_label2id, id2type_label, num_types = load_allusion_dict()
    dict_size = len(allusion_dict)
    
    # 创建模型时使用正确的参数大小
    model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size).to(device)
    
    # 现在加载参数应该不会出错
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    # 预处理特征和映射文件路径
    features_path = os.path.join(DATA_DIR, 'allusion_features_large_dict.pt')
    mapping_path = os.path.join(DATA_DIR, 'allusion_mapping_large_dict.json')
    
    # 准备数据集
    test_dataset = PoetryNERDataset(
        TEST_PATH, tokenizer, MAX_SEQ_LEN,
        type_label2id=type_label2id,
        id2type_label=id2type_label,
        task='position',
        features_path=features_path,
        mapping_path=mapping_path
        )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    
    # 分析错误
    mistakes, high_loss_cases, allusion_confusion = analyze_mistakes(model, test_dataloader, device)
    
    # 1. 打印高损失案例
    print("\n=== 高损失案例 Top 10 ===")
    for i in range(min(10, len(high_loss_cases))):
        _, _, _, case = heapq.heappop(high_loss_cases)
        print(f"\n案例 {i+1} (Loss: {case['loss']:.4f}):")
        print(f"原文: {case['text']}")
        print(f"正确标注: {format_labels(case['text'], case['true_labels'])}")
        print(f"模型预测: {format_labels(case['text'], case['pred_labels'])}")
        print(f"正确典故: {', '.join(case['true_allusion'])}")
        print(f"预测典故: {', '.join(case['pred_allusion'])}")
    
    # 2. 打印错误类型分析
    print("\n=== 错误类型分析 ===")
    for mistake_type, cases in mistakes.items():
        print(f"\n{mistake_type} 错误数量: {len(cases)}")
        print("典型案例:")
        for i, case in enumerate(cases[:5]):
            print(f"\n案例 {i+1}:")
            print(f"原文: {case['text']}")
            print(f"正确标注: {format_labels(case['text'], case['true_labels'])}")
            print(f"模型预测: {format_labels(case['text'], case['pred_labels'])}")
            print(f"错误位置字符: {case['char']}")
    
    # 3. 打印典故混淆分析
    print("\n=== 典故混淆分析 ===")
    for true_allusion, confusions in allusion_confusion.items():
        if true_allusion == 'NONE':  # 跳过空典故
            continue
        print(f"\n正确典故: {true_allusion}")
        sorted_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)
        print("常见混淆:")
        for pred_allusion, count in sorted_confusions[:5]:
            print(f"  - 预测为: {pred_allusion}, 次数: {count}")

if __name__ == '__main__':
    main()
