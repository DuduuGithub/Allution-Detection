import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from model.bert_crf import prepare_sparse_features
from model.train import load_allusion_dict
from model.config import DATA_DIR, SAVE_DIR
import json

def create_and_save_sentence_mappings(source_path, output_path):
    """
    创建并保存句子映射
    Args:
        source_path: 数据集路径
        output_path: 保存路径
    Returns:
        dict: sentence_to_id映射
    """
    sentences = set()

    # 读取数据文件
    df = pd.read_csv(source_path, sep='\t')
    sentences.update(df['sentence'].unique())
    
    # 创建固定的句子到ID的映射
    sentence_to_id = {sent: idx for idx, sent in enumerate(sorted(sentences))}
    id_to_sentence = {idx: sent for sent, idx in sentence_to_id.items()}
    
    # 保存映射
    mappings = {
        'sentence_to_id': sentence_to_id,
        'id_to_sentence': id_to_sentence,
        'total_sentences': len(sentence_to_id)
    }
    
    # 使用JSON格式保存映射（更容易查看和调试）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)
        
    print(f"Sentence mappings saved to {output_path}")
    print(f"Total unique sentences: {len(sentence_to_id)}")
    
    return sentence_to_id

def preprocess_all_features(sentence_to_id, features_output_path):
    """预处理所有句子的特征并保存"""
    # 加载典故字典
    allusion_dict, _, _, _ = load_allusion_dict()
    
    # 存储所有特征
    all_features = {}
    
    # 逐个处理句子
    sentences = list(sentence_to_id.keys())
    
    for sent in tqdm(sentences, desc="Preprocessing features"):
        # 生成特征
        features = prepare_sparse_features([sent], allusion_dict)
        
        # 获取句子ID
        sent_id = sentence_to_id[sent]
        
        # 提取单个句子的特征
        all_features[sent_id] = {
            'indices': features['indices'][0].to(torch.int16),
            'values': features['values'][0].half(),
            'active_counts': features['active_counts'][0].to(torch.int16)
        }
        
    
    # 保存特征
    torch.save(all_features, features_output_path)
    
    # 计算内存使用
    total_size = os.path.getsize(features_output_path) / (1024 * 1024)  # MB
    print(f"\nFeatures saved to {features_output_path}")
    print(f"Total file size: {total_size:.2f} MB")

def load_sentence_mappings(mapping_path):
    """
    加载句子映射
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    return mappings['sentence_to_id'], mappings['id_to_sentence']

def load_features(features_path):
    """
    加载预处理的特征
    """
    return torch.load(features_path)

if __name__ == "__main__":
    # 数据文件路径
    source_path = os.path.join(DATA_DIR, 'all_data.csv')
    
    # 映射文件和特征文件的保存路径
    mapping_path = os.path.join(DATA_DIR, 'allusion_mapping.json')
    features_path = os.path.join(DATA_DIR, 'allusion_features.pt')
    
    # 创建并保存句子映射
    sentence_to_id = create_and_save_sentence_mappings(
        source_path, mapping_path
    )
    
    # 预处理并保存特征
    preprocess_all_features(sentence_to_id, features_path)