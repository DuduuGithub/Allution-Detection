import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from tqdm import tqdm
from evaluate_mini_sample import analyze_type_errors_and_training_stats

def load_model_and_tokenizer(model_path):
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer

def predict_type(text, model, tokenizer):
    """预测单个文本的类型"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=-1).item()
    
    return model.config.id2label[pred_label]

def main():
    # 设置路径
    model_path = "path/to/your/model"  # 请替换为实际的模型路径
    test_file = "path/to/test.json"    # 请替换为实际的测试文件路径
    training_file = "path/to/train.tsv" # 请替换为实际的训练文件路径
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # 加载测试数据
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 收集错误案例
    error_cases = {'type': []}
    
    # 处理每个测试样本
    for sample in tqdm(test_data, desc="Processing test samples"):
        text = sample['text']
        true_type = sample['type']
        
        # 预测类型
        pred_type = predict_type(text, model, tokenizer)
        
        # 如果预测错误，添加到错误案例中
        if pred_type != true_type:
            error_cases['type'].append({
                'text': text,
                'type_true': [true_type],
                'type_pred': [pred_type]
            })
    
    # 分析错误案例
    analysis_results = analyze_type_errors_and_training_stats(error_cases, training_file)
    
    print(f"\nTotal samples: {len(test_data)}")
    print(f"Error cases: {len(error_cases['type'])}")
    print(f"Accuracy: {(len(test_data) - len(error_cases['type'])) / len(test_data):.2%}")

if __name__ == "__main__":
    main() 