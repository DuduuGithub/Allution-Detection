import os

# 模型配置
MODEL_NAME = 'guwenbert-large'
BERT_MODEL_PATH = os.path.join('model', MODEL_NAME)
POSITION_EPOCHS = 15
TYPE_EPOCHS = 15
LEARNING_RATE = 2e-5

# 数据配置
MAX_SEQ_LEN = 128
BATCH_SIZE = 1


# 聚类配置
min_samples_size=2
OPTIMAL_EPS = 0.3824829931972789

# 路径配置
DATA_DIR = 'data'
SAVE_DIR = 'output'
ALLUSION_TYPES_PATH = 'data/updated_典故的异性数据.csv'

# 标签配置
POSITION_LABELS = {
    'O': 0,
    'B': 1,
    'I': 2
}
