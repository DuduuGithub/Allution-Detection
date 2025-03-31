'''
    测试字典特征的结果
'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.config import DATA_DIR
from process.process_dict_features import load_sentence_mappings, load_features
from model.train import load_allusion_dict

def test_features():
    # 加载映射和特征文件
    mapping_path = os.path.join(DATA_DIR, 'allusion_mapping_large_dict.json')
    features_path = os.path.join(DATA_DIR, 'allusion_features_large_dict.pt')
    
    sentence_to_id, id_to_sentence = load_sentence_mappings(mapping_path)
    all_features = load_features(features_path)
    
    # 加载典故词典用于反查
    allusion_dict, _, _, _ = load_allusion_dict()
    allusion_to_id = {name: idx for idx, name in enumerate(allusion_dict.keys())}
    id_to_allusion = {idx: name for name, idx in allusion_to_id.items()}
    
    # 测试一些样例诗句
    test_sentences = [
        "晨严九折度，暮戒六军行。",
        "郢曲怜公子，吴州忆伯鸾。",
        "自怜非剧孟，何以佐良图。"
    ]
    
    for sentence in test_sentences:
        print(f"\n=== 测试诗句: {sentence} ===")
        
        # 获取该句子的ID和特征
        if sentence not in sentence_to_id:
            print("句子不在映射中！")
            continue
            
        sent_id = sentence_to_id[sentence]
        features = all_features[sent_id]
        
        # 转换数据类型
        indices = features['indices'].long()
        values = features['values'].float()
        active_counts = features['active_counts'].long()
        
        print(f"特征形状: indices={indices.shape}, values={values.shape}, active_counts={active_counts.shape}")
        print(f"indices: {indices}")
        print(f"values: {values}")
        print(f"active_counts: {active_counts}")
        # 打印完整的特征矩阵
        print("\n完整特征矩阵:")
        print("位置\t字符\t活跃数\t典故ID\t\t相似度")
        print("-" * 80)
        
        for pos in range(len(indices)):
            char = sentence[pos] if pos < len(sentence) else "PAD"
            count = active_counts[pos].item()
            
            # 打印该位置的所有特征
            if count > 0:
                # 第一行显示位置信息
                print(f"{pos}\t{char}\t{count}", end='\t')
                # 显示第一个典故
                allusion_id = indices[pos, 0].item()
                similarity = values[pos, 0].item()
                allusion_name = id_to_allusion[allusion_id] if allusion_id in id_to_allusion else "Unknown"
                print(f"{allusion_id}({allusion_name})\t{similarity:.3f}")
                
                # 显示剩余的典故（如果有的话）
                for i in range(1, count):
                    allusion_id = indices[pos, i].item()
                    similarity = values[pos, i].item()
                    allusion_name = id_to_allusion[allusion_id] if allusion_id in id_to_allusion else "Unknown"
                    print(f"\t\t\t{allusion_id}({allusion_name})\t{similarity:.3f}")
            else:
                print(f"{pos}\t{char}\t{count}\t-\t-")

if __name__ == "__main__":
    test_features() 