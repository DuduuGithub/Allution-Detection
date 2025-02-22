from transformers import BertTokenizer
from poetry_dataset import PoetryNERDataset, load_allusion_types
from config import BERT_MODEL_PATH, TEST_PATH, ALLUSION_TYPES_PATH

def test_type_loading():
    print("Testing type loading...")
    
    # 1. 直接从 allusion_types.txt 加载类型
    type_label2id, id2type_label = load_allusion_types(ALLUSION_TYPES_PATH)
    print(f"\nFrom allusion_types.txt:")
    print(f"Number of types: {len(type_label2id)}")
    print(f"First 5 types: {list(type_label2id.keys())[:5]}")
    
    # 2. 通过数据集加载类型
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    dataset = PoetryNERDataset(TEST_PATH, tokenizer, max_len=128, task='type')
    
    print(f"\nFrom dataset:")
    print(f"Number of types in dataset: {len(dataset.type_label2id)}")
    print(f"First 5 types in dataset: {list(dataset.type_label2id.keys())[:5]}")
    
    # 3. 验证两种方式加载的类型是否一致
    print(f"\nVerification:")
    print(f"Types are identical: {type_label2id == dataset.type_label2id}")
    
    if type_label2id != dataset.type_label2id:
        # 找出差异
        types_from_file = set(type_label2id.keys())
        types_from_dataset = set(dataset.type_label2id.keys())
        
        print("\nTypes in file but not in dataset:")
        print(list(types_from_file - types_from_dataset)[:5])
        
        print("\nTypes in dataset but not in file:")
        print(list(types_from_dataset - types_from_file)[:5])

if __name__ == "__main__":
    test_type_loading() 