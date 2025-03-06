import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import BertTokenizer
from model.bert_crf import AllusionBERTCRF
from model.train import load_allusion_dict
from model.config import BERT_MODEL_PATH, MAX_SEQ_LEN, SAVE_DIR
from model.bert_crf import prepare_sparse_features

def load_models():
    """加载预训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    # 加载典故词典以获取类型数量
    allusion_dict, _, _, num_types = load_allusion_dict()
    dict_size = len(allusion_dict)
    
    # 创建两个模型实例
    position_model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size=dict_size).to(device)
    type_model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size=dict_size).to(device)
    
    # 分别加载两个模型的参数
    position_checkpoint = torch.load(f'{SAVE_DIR}/best_model_position.pt', map_location=device)
    position_model.load_state_dict(position_checkpoint['model_state_dict'])
    position_model.eval()
    
    type_checkpoint = torch.load(f'{SAVE_DIR}/best_model_type.pt', map_location=device)
    type_model.load_state_dict(type_checkpoint['model_state_dict'])
    type_model.eval()
    
    return position_model, type_model, tokenizer, device

def predict_allusion(text, position_model, type_model, tokenizer, device):
    """
    预测一句诗中的典故及其类型
    
    Args:
        text: 输入的诗句
        position_model: 位置识别模型
        type_model: 类型识别模型
        tokenizer: BERT tokenizer
        device: 计算设备
    
    Returns:
        list: [(start_pos, end_pos, [(type1, prob1), (type2, prob2), ...]), ...]
    """
    # 准备输入数据
    seq_len = len(text) + 2  # 加2是因为[CLS]和[SEP]
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=seq_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 获取字典特征
    allusion_dict, _, id2type_label, _ = load_allusion_dict()
    dict_features = prepare_sparse_features([text], allusion_dict)
    dict_features = {
        'indices': dict_features['indices'].to(device),
        'values': dict_features['values'].to(device),
        'active_counts': dict_features['active_counts'].to(device)
    }
    
    # 1. 位置识别
    with torch.no_grad():
        position_predictions = position_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dict_features=dict_features,
            task='position'
        )
    
    print("Raw position predictions:", position_predictions)
    
    # 如果position_predictions是嵌套列表，获取内部的列表
    if isinstance(position_predictions, list):
        position_predictions = position_predictions[0]
    # 如果是张量，转换为列表
    elif torch.is_tensor(position_predictions):
        position_predictions = position_predictions[0].cpu().tolist()
    
    # 2. 提取典故位置
    allusion_positions = []
    i = 0
    while i < len(position_predictions):
        if position_predictions[i] == 1:  # B标签
            start = i
            end = i
            # 寻找典故结束位置
            for j in range(i + 1, len(position_predictions)):
                if position_predictions[j] == 2:  # I标签
                    end = j
                else:
                    break
            allusion_positions.append((start, end))
            i = end + 1
        else:
            i += 1
    
    print('position_predictions:', position_predictions)
    print('allusion_positions:', allusion_positions)
    
    # 3. 对每个位置进行类型识别
    results = []
    for start, end in allusion_positions:
        # 准备类型识别的输入
        with torch.no_grad():
            # 获取预测结果
            target_positions = torch.tensor([[start + 1, end + 1]]).to(device)  # +1 因为有CLS token
            pred_result = type_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dict_features=dict_features,
                task='type',
                target_positions=target_positions
            )
        
        # 获取前5个预测的类型和概率
        predictions = pred_result['predictions'][0]  # 第一个样本的预测
        probabilities = pred_result['probabilities'][0]  # 第一个样本的概率
        
        # 将预测ID转换为类型名称，并与概率配对
        top5_results = []
        for pred_id, prob in zip(predictions, probabilities):
            pred_type = id2type_label[pred_id.item()]
            top5_results.append((pred_type, prob.item()))
        
        results.append((start, end, top5_results))
    
    return results

def main():
    # 加载模型
    position_model, type_model, tokenizer, device = load_models()
    
    # 测试用例
    test_poems = [
        # "桃源避秦人不见，武陵渔父独知处。",
        # "一入石渠署，三闻宫树蝉。",
        # '穷途行泣玉，愤路未藏金。',
        # '莱子多嘉庆，陶公得此生。',
        # '扬风非赠扇，易俗是张琴。',
        # '尚激抟溟势，期君借北风。',
        # '恐入壶中住，须传肘后方。'
        '潘郎作赋年，陶令辞官后。'
        
    ]
    
    for poem in test_poems:
        print(f"\n诗句: {poem}")
        results = predict_allusion(poem, position_model, type_model, tokenizer, device)
        
        print("预测结果:")
        for start, end, predictions in results:
            allusion_text = poem[start:end+1]
            print(f"\n典故: {allusion_text} ({start}-{end})")
            for pred_type, prob in predictions:
                print(f"  - {pred_type}: {prob:.3f}")

if __name__ == "__main__":
    main()