import numpy as np
from collections import Counter
from torch.utils.data import WeightedRandomSampler

class DataBalancer:
    def __init__(self, dataset, strategy='weighted'):
        self.dataset = dataset
        self.strategy = strategy
        
    def get_sampler(self):
        """获取平衡的采样器"""
        if self.dataset.task == 'type':
            # 获取所有类型标签
            labels = [item['target_type'] for item in self.dataset.data]
        else:
            # 对于位置识别任务，统计B和I标签
            labels = []
            for item in self.dataset.data:
                labels.extend([l for l in item['position_labels'] if l != 0])  # 排除O标签
                
        # 计算类别分布
        label_counter = Counter(labels)
        print("\n类别分布统计:")
        for label, count in label_counter.items():
            if self.dataset.task == 'type':
                label_name = self.dataset.id2type_label.get(label, str(label))
            else:
                label_name = 'B' if label == 1 else 'I'
            print(f"类别 {label_name}: {count} 样本")
        
        # 计算权重
        weights = [1.0 / label_counter[label] for label in labels]
        weights = torch.FloatTensor(weights)
        
        # 创建采样器
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        
        return sampler 