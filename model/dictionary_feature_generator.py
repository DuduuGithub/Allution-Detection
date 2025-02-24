from scipy.sparse import csr_matrix
import numpy as np
from difflib import SequenceMatcher

class DictionaryFeatureGenerator:
    def __init__(self, allusion_dict):
        """初始化特征生成器
        
        Args:
            allusion_dict: 典故词典，格式为 {
                '典故1': ['异形词1', '异形词2', ...],
                '典故2': ['异形词3', '异形词4', ...],
                ...
            }
        """
        self.allusion_dict = allusion_dict
        # 为每个典故分配唯一的索引
        self.allusion_to_idx = {allusion: idx for idx, allusion in enumerate(allusion_dict.keys())}
        
    def string_similarity(self, str1, str2):
        return SequenceMatcher(None, str1, str2).ratio()
    
    def generate_sparse_features(self, text, similarity_threshold=0.8):
        """生成稀疏特征向量
        
        Args:
            text: 输入文本
            similarity_threshold: 相似度阈值
            
        Returns:
            sparse_vector: CSR格式的稀疏矩阵
            matched_info: 匹配信息
        """
        # 存储非零元素信息
        data = []       # 非零值列表
        indices = []    # 非零元素的列索引
        matched_info = []
        
        # 对每个典故进行匹配
        for allusion, variants in self.allusion_dict.items():
            allusion_idx = self.allusion_to_idx[allusion]
            
            # 检查文本是否匹配任何变体
            for variant in variants:
                similarity = self.string_similarity(text, variant)
                if similarity >= similarity_threshold:
                    data.append(1.0)  # 匹配时设置为1
                    indices.append(allusion_idx)
                    matched_info.append({
                        'allusion': allusion,
                        'matched_variant': variant,
                        'similarity': similarity
                    })
                    break  # 找到一个匹配就跳出内层循环
        
        # 创建CSR格式稀疏矩阵
        if not indices:
            # 如果没有匹配，返回全零稀疏向量
            return csr_matrix((1, len(self.allusion_to_idx)), dtype=np.float32), []
        
        # 构造CSR矩阵的三个关键数组
        data = np.array(data)
        indices = np.array(indices)
        indptr = np.array([0, len(indices)])  # 行指针，只有一行所以很简单
        
        # 创建稀疏矩阵 (1 x num_allusions)
        sparse_vector = csr_matrix((data, indices, indptr),
                                 shape=(1, len(self.allusion_to_idx)))
        
        return sparse_vector, matched_info

    def explain_features(self, sparse_vector):
        """解释稀疏向量中的非零特征
        
        Args:
            sparse_vector: CSR格式的稀疏矩阵
            
        Returns:
            list: 激活的典故列表
        """
        active_features = []
        # 获取非零元素的索引
        nonzero_indices = sparse_vector.indices
        
        # 反向映射得到典故名称
        idx_to_allusion = {idx: allusion for allusion, idx in self.allusion_to_idx.items()}
        
        for idx in nonzero_indices:
            active_features.append(idx_to_allusion[idx])
            
        return active_features

# 使用示例
if __name__ == "__main__":
    # 示例典故词典
    example_dict = {
        "三径": ["开三径", "三径", "开竹径"],
        "羊求": ["羊求", "求仲", "羊仲", "求羊径"],
        "蒋诩": ["蒋诩径", "蒋生径"]
    }
    
    # 初始化生成器
    generator = DictionaryFeatureGenerator(example_dict)
    
    # 生成特征
    text = "诗人开三径而居"
    sparse_features, matches = generator.generate_sparse_features(text)
    
    # 输出结果
    print("Sparse vector shape:", sparse_features.shape)
    print("Non-zero elements:", sparse_features.nonzero()[1])
    print("Matched allusions:", generator.explain_features(sparse_features))
    print("Match details:", matches) 