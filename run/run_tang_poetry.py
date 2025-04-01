import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import BertTokenizer
from model.bert_crf import AllusionBERTCRF, prepare_sparse_features
from model.train import load_allusion_dict
from model.config import (
    BERT_MODEL_PATH, MAX_SEQ_LEN,
    BATCH_SIZE, DATA_DIR
)

def load_model(model_name):
    """加载预训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    # 加载典故词典以获取类型数量
    allusion_dict, _, id2type_label, num_types = load_allusion_dict()
    dict_size = len(allusion_dict)
    
    # 创建模型实例
    model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size=dict_size).to(device)
    
    # 加载模型参数
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAVE_DIR = os.path.join(PROJECT_ROOT, 'trained_result')
    
    checkpoint = torch.load(f'{SAVE_DIR}/{model_name}.pt', map_location=device)
    
    # 处理模型状态字典
    state_dict = checkpoint['model_state_dict']
    for key in list(state_dict.keys()):
        if key not in model.state_dict():
            print(f"Removing unexpected key: {key}")
            del state_dict[key]
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, tokenizer, device, allusion_dict, id2type_label

def prepare_batch_data(texts, tokenizer, allusion_dict):
    """
    准备批量数据，参考 PoetryNERDataset 的 collate_fn 函数
    """
    # 获取最大长度
    max_text_len = max(len(text) for text in texts)
    
    # 准备batch数据的列表
    batch_texts = []
    batch_input_ids = []
    batch_attention_mask = []
    indices_list = []
    values_list = []
    active_counts_list = []
    
    # 处理每个文本
    for text in texts:
        batch_texts.append(text)
        
        # BERT tokenization
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_text_len + 2,  # +2 for [CLS] and [SEP]
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 获取input_ids和attention_mask
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        
        # 处理字典特征
        text_features = prepare_sparse_features([text], allusion_dict)
        
        # 获取当前序列长度
        seq_len = len(input_ids)
        
        # 处理字典特征的维度
        indices = text_features['indices'].squeeze(0)[:seq_len]
        values = text_features['values'].squeeze(0)[:seq_len]
        active_counts = text_features['active_counts'].squeeze(0)[:seq_len]
        
        # 补全到最大长度（保持[CLS]位置为0）
        if indices.size(0) < seq_len:
            pad_len = seq_len - indices.size(0)
            indices = torch.cat([indices, torch.zeros((pad_len, 5), dtype=torch.long)], dim=0)
            values = torch.cat([values, torch.zeros((pad_len, 5), dtype=torch.float)], dim=0)
            active_counts = torch.cat([active_counts, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        
        indices_list.append(indices)
        values_list.append(values)
        active_counts_list.append(active_counts)
    
    # 堆叠所有张量
    return {
        'texts': batch_texts,
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
        'dict_features': {
            'indices': torch.stack(indices_list),
            'values': torch.stack(values_list),
            'active_counts': torch.stack(active_counts_list)
        }
    }

def predict_batch(texts, model, tokenizer, device, allusion_dict, id2type_label):
    """批量预测多个文本中的典故"""
    # 准备批量数据
    batch_data = prepare_batch_data(texts, tokenizer, allusion_dict)
    # 将数据移到设备上
    input_ids = batch_data['input_ids'].to(device)
    attention_mask = batch_data['attention_mask'].to(device)
    dict_features = {
        'indices': batch_data['dict_features']['indices'].to(device),
        'values': batch_data['dict_features']['values'].to(device),
        'active_counts': batch_data['dict_features']['active_counts'].to(device)
    }
    
    batch_results = []
    
    with torch.no_grad():
        # 位置识别
        position_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dict_features=dict_features,
            train_mode=False,
            task='position'
        )
        
        position_predictions = position_outputs['position_predictions']
        # 处理每个样本
        for batch_idx, (text, position_pred) in enumerate(zip(texts, position_predictions)):
            text_results = []
            
            i = 0
            while i < len(position_pred):
                if position_pred[i] == 1:  # B标签
                    start = i
                    end = i
                    # 寻找典故结束位置
                    for j in range(i + 1, len(position_pred)):
                        if position_pred[j] == 2:  # I标签
                            end = j
                        else:
                            break
                    
                    # 准备类型识别的输入
                    target_positions = torch.tensor([[[start+1, end+1]]], device=device)
                    
                    # 类型识别
                    type_outputs = model(
                        input_ids=input_ids[batch_idx:batch_idx+1],
                        attention_mask=attention_mask[batch_idx:batch_idx+1],
                        dict_features={
                            'indices': dict_features['indices'][batch_idx:batch_idx+1],
                            'values': dict_features['values'][batch_idx:batch_idx+1],
                            'active_counts': dict_features['active_counts'][batch_idx:batch_idx+1]
                        },
                        target_positions=target_positions,
                        train_mode=False,
                        task='type'
                    )
                    
                    # 处理类型预测结果
                    type_predictions = type_outputs['type_predictions']
                    if type_predictions:
                        _, _, type_probs = type_predictions[0]
                        text_results.append((start, end, type_probs)) 
                    i = end
                else:
                    i += 1
            
            batch_results.append(text_results)
    
    return batch_results

def parse_tang_poetry(file_path):
    """解析唐诗文本文件"""
    poems = []
    
    # 直接使用 GBK 编码读取
    with open(file_path, 'r', encoding='gbk') as f:
        content = f.read()
        print('使用 GBK 编码读取成功')

    # 直接按 ◎卷. 分割所有诗
    poems_texts = content.split('◎卷.')
    
    current_volume = None
    # 处理第一段文本（可能包含卷号信息）
    volume_match = re.search(r'第([一二三四五六七八九十百千]+)卷|第(\d+)卷', poems_texts[0])
    if volume_match:
        current_volume = volume_match.group(1) or volume_match.group(2)
        print(f"找到起始卷号: {current_volume}")
    
    # 从第二段开始处理（第一段只包含卷号信息）
    for poem_text in poems_texts[1:]:
        if not poem_text.strip():
            continue
            
        # 检查是否有新的卷号
        volume_match = re.search(r'第([一二三四五六七八九十百千]+)卷|第(\d+)卷', poem_text)
        if volume_match:
            current_volume = volume_match.group(1) or volume_match.group(2)
            print(f"切换到新卷号: {current_volume}")
            
        # 尝试两种格式匹配
        sequence_match = re.match(r'(\d+)【(.+?)】(\w+)\n((?:.*?\n)*)', poem_text)
        if not sequence_match:
            # 尝试匹配纯数字序号格式
            sequence_match = re.match(r'(\d+)(.*?)\n((?:.*?\n)*)', poem_text)
            if not sequence_match:
                print(f"无法匹配的诗文本: {poem_text[:100]}...")  # 打印前100个字符用于调试
                continue
            
            # 处理纯数字序号格式
            sequence_number = sequence_match.group(1)
            title_author = sequence_match.group(2).strip()
            content = sequence_match.group(3)
            
            # 从标题中提取作者（如果有）
            title_author_match = re.match(r'【(.+?)】(\w+)', title_author)
            if title_author_match:
                title = title_author_match.group(1)
                author = title_author_match.group(2)
            else:
                title = title_author
                author = "佚名"
        else:
            # 处理带【】的格式
            sequence_number, title, author, content = sequence_match.groups()
        
        if not current_volume:
            print("警告: 未找到当前卷号")
            continue
        
        # 处理诗的内容
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        # 只保留实际的诗句（去掉空行和其他标记）
        poem_lines = [line for line in lines if not line.startswith('--')]
        
        poems.append({
            'volume_number': current_volume,
            'sequence_number': sequence_number,  # 保存序号（卷.后的数字）
            'title': title,
            'author': author,
            'content': poem_lines
        })
        
        if len(poems) % 100 == 0:
            print(f"已解析 {len(poems)} 首诗")
    
    print(f"共解析出 {len(poems)} 首诗")
    return poems

def save_single_poem_result(poem, output_file, mode='a', current_volume=None):
    """
    保存单首诗的分析结果到文件
    mode: 'a' 为追加模式，'w' 为覆写模式
    current_volume: 当前正在处理的卷号，用于判断是否需要输出卷标题
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_file, mode, encoding='utf-8') as f:
        # 只在卷号变化时才写入卷标题
        if current_volume != poem['volume_number']:
            f.write(f"\n第{poem['volume_number']}卷\n")
        
        # 写入诗歌信息，使用《》表示标题
        f.write(f"    ◎卷.{poem['sequence_number']}《{poem['title']}》{poem['author']}\n")
        
        # 写入诗句，并在典故处添加标注
        for line, allusions in zip(poem['content'], poem['allusions']):
            marked_line = line
            for allusion in sorted(allusions, key=lambda x: x['position'][0], reverse=True):
                start, end = allusion['position']
                marked_line = (marked_line[:start] + 
                             f"「{marked_line[start:end+1]}」" + 
                             marked_line[end+1:])
            f.write(f"    {marked_line}\n")
        
        # 写入典故说明
        if any(poem['allusions']):  # 如果有典故
            f.write("\n    典故：\n")
            allusion_count = 0
            for line_idx, allusions in enumerate(poem['allusions']):
                for allusion in allusions:
                    allusion_count += 1
                    f.write(f"    {allusion_count}. {allusion['text']}: "
                          f"{allusion['type']} (置信度: {allusion['probability']:.3f})\n")
        else:
            f.write("\n    无典故\n")
        
        f.write("\n")  # 诗与诗之间空一行
        
        return poem['volume_number']  # 返回当前卷号

def process_single_poem(poem, model, tokenizer, device, allusion_dict, id2type_label):
    """处理单首诗并识别典故"""
    # 批量处理诗句
    results = predict_batch(poem['content'], model, tokenizer, device, allusion_dict, id2type_label)
    
    # 保存典故结果
    allusions_list = []
    for line, line_results in zip(poem['content'], results):
        line_allusions = []
        if line_results:
            for start, end, predictions in line_results:
                allusion_text = line[start:end+1]
                best_pred = max(predictions, key=lambda x: x[1])
                type_name = id2type_label[best_pred[0]]
                line_allusions.append({
                    'text': allusion_text,
                    'position': (start, end),
                    'type': type_name,
                    'probability': best_pred[1]
                })
        allusions_list.append(line_allusions)
    
    # 将典故信息添加到诗歌数据中
    poem['allusions'] = allusions_list
    return poem

def main():
    # 获取项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("开始加载模型...")
    # 加载模型
    model, tokenizer, device, allusion_dict, id2type_label = load_model('output_jointly_train_normalize_loss/best_model_3.17.22.09')
    print("模型加载完成")
    
    # 读取唐诗文件
    input_file = os.path.join(PROJECT_ROOT, "全唐诗", "《全唐诗》作者：曹寅.txt")
    print("\n开始解析唐诗文件...")
    poems = parse_tang_poetry(input_file)
    
    # 设置输出文件
    output_file = os.path.join(PROJECT_ROOT, "全唐诗", "典故分析结果.txt")
    
    print(f"\n开始处理诗歌，共 {len(poems)} 首...")
    current_volume = None
    
    for i, poem in enumerate(poems, 1):
        print(f"\n{'='*50}")
        print(f"正在处理第 {i}/{len(poems)} 首诗")
        print(f"卷号: {poem['volume_number']}")
        print(f"序号: 卷.{poem['sequence_number']}")
        print(f"题目: {poem['title']}")
        print(f"作者: {poem['author']}")
        
        # 处理诗歌
        processed_poem = process_single_poem(poem, model, tokenizer, device, allusion_dict, id2type_label)
        
        # 追加保存结果，并更新当前卷号
        current_volume = save_single_poem_result(processed_poem, output_file, mode='a', current_volume=current_volume)
        
        # 显示典故数量
        allusion_count = sum(len(allusions) for allusions in processed_poem['allusions'])
        print(f"发现典故: {allusion_count} 个")
        print(f"已保存到: {output_file}")
        
        # 每处理10首诗显示一次进度百分比
        if i % 10 == 0:
            progress = (i / len(poems)) * 100
            print(f"\n当前进度: {progress:.2f}%")
    
    print(f"\n{'='*50}")
    print("所有诗歌处理完成！")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 