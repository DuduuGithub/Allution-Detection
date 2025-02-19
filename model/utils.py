def load_allusion_types(file_path):
    """从文件加载典故类型映射"""
    type_label2id = {}
    id2type_label = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        types = [line.strip() for line in f.readlines() if line.strip()]
        
    # 创建双向映射
    for idx, type_label in enumerate(sorted(types)):
        type_label2id[type_label] = idx
        id2type_label[idx] = type_label
    
    return type_label2id, id2type_label 