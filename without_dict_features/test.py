from transformers import BertTokenizer
import json
import os

def test_tokenizer(text_samples):
    """测试BERT tokenizer对古诗的分词效果"""
    try:
        # 获取当前文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取guwenbert-large的绝对路径
        model_path = os.path.join(os.path.dirname(current_dir), 'model', 'guwenbert-large')
        
        print(f"尝试加载模型，路径: {model_path}")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        print(f"成功从 {model_path} 加载tokenizer")
        
        results = []
        for text in text_samples:
            # 获取tokenizer的结果
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text)
            decoded_text = tokenizer.decode(token_ids)
            
            result = {
                "原文": text,
                "分词结果": tokens,
                "token_ids": token_ids,
                "重建文本": decoded_text,
                "token数量": len(tokens)
            }
            results.append(result)
            
            # 打印详细结果
            print("\n" + "="*50)
            print(f"原文: {text}")
            print(f"分词结果: {' '.join(tokens)}")
            print(f"Token IDs: {token_ids}")
            print(f"重建文本: {decoded_text}")
            print(f"Token数量: {len(tokens)}")
        
        # 保存结果到文件
        with open('tokenizer_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"加载模型时出错: {e}")

def main():
    # 测试样例，包含不同类型的古诗
    test_samples = [
        "晋太元中，武陵人捕鱼为业。"
    ]
    
    # 测试特殊情况
    special_cases = [
        "「」『』？！。，",  # 标点符号
        "子曰：「学而时习之，不亦说乎？」",  # 文言文
        "床前[UNK]月光",  # 包含未知字符
        "床　前　明　月　光",  # 全角空格
    ]
    
    test_samples.extend(special_cases)
    test_tokenizer(test_samples)

if __name__ == "__main__":
    main()
