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
    model = AllusionBERTCRF(BERT_MODEL_PATH, num_types, dict_size=dict_size).to(device)
    
    # 分别加载两个模型的参数
    checkpoint = torch.load(f'{SAVE_DIR}/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model,tokenizer, device

def predict_allusion(text, model, tokenizer, device):
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
        position_predictions_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dict_features=dict_features,
            train_mode=False,
            task='position'
        )
    
    print("Raw position predictions output:", position_predictions_output)
    position_predictions=position_predictions_output['position_predictions']
    # 提取典故位置
    allusion_positions = []
    i = 0
    while i < len(position_predictions[0]):
        if position_predictions[0][i] == 1:  # B标签
            start = i
            end = i
            # 寻找典故结束位置
            for j in range(i + 1, len(position_predictions[0])):
                if position_predictions[0][j] == 2:  # I标签
                    end = j
                else:
                    break
            allusion_positions.append((start, end))
            i = end + 1
        else:
            i += 1

    print('position_predictions:', position_predictions[0])
    print('allusion_positions:', allusion_positions)

    # 处理 target_positions
    if len(allusion_positions) == 0:
        # 如果没有检测到典故位置，填充默认值
        target_positions = torch.tensor([[[0, 0]]], device=device)
    else:
        # 只取第一个典故位置
        start, end = allusion_positions[0]
        pos_tensor = torch.tensor([[start + 1, end + 1]], device=device)
        # 调整为 [1, 1, 2] 的形状
        target_positions = pos_tensor.unsqueeze(0)  # 或者使用 pos_tensor.view(1, 1, 2)
    print('target_positions:', target_positions)
    
    type_pred_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        dict_features=dict_features,
        target_positions=target_positions,
        train_mode=False,
        task='type'
    )
    
    # 替换 type_id 为 pred_type
    allusion_list = []
    for start, end, type_list in type_pred_outputs['type_predictions']:
        new_type_list = []
        for type_id, prob in type_list:
            pred_type = id2type_label[type_id]  # 替换 type_id
            new_type_list.append((pred_type, prob))
        allusion_list.append((start, end, new_type_list))
    
    return allusion_list

def main():
    # 加载模型
    model, tokenizer, device = load_models()
    
    # 测试用例
    test_poems = [
        "桃源避秦人不见，武陵渔父独知处。",
        "一入石渠署，三闻宫树蝉。",
        # '穷途行泣玉，愤路未藏金。',
        # '莱子多嘉庆，陶公得此生。',
        # '扬风非赠扇，易俗是张琴。',
        # '尚激抟溟势，期君借北风。',
        # '恐入壶中住，须传肘后方。',
        # '潘郎作赋年，陶令辞官后。',
        # '唯是贾生先恸哭，不堪天意重阴云。',
        # '梳洗凭张敞，乘骑笑稚恭。'
        
    ]
    
    for poem in test_poems:
        print(f"\n诗句: {poem}")
        results = predict_allusion(poem, model,tokenizer, device)
        
        print("预测结果:")
        for start, end, predictions in results:
            allusion_text = poem[start:end+1]
            print(f"\n典故: {allusion_text} ({start}-{end})")
            for pred_type, prob in predictions:
                print(f"  - {pred_type}: {prob:.3f}")

if __name__ == "__main__":
    main()