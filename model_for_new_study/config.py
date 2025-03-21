import os

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型配置
MODEL_NAME = 'guwenbert-large'
BERT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', MODEL_NAME)
EPOCHS=25
LEARNING_RATE = 2e-5

# 数据配置
MAX_SEQ_LEN = 128
BATCH_SIZE = 16


# 聚类配置
min_samples_size=2
OPTIMAL_EPS = 0.3824829931972789

# 路径配置
DATA_DIR = os.path.join(PROJECT_ROOT, 'data_for_new_study')
# SAVE_DIR = os.path.join(PROJECT_ROOT, 'output_strictly')
# SAVE_DIR = os.path.join(PROJECT_ROOT, 'output_train_together')
# SAVE_DIR = os.path.join(PROJECT_ROOT, 'output_jointly_train')
# SAVE_DIR = os.path.join(PROJECT_ROOT, 'output_jointly_train_normalize_loss')
SAVE_DIR = os.path.join(PROJECT_ROOT, 'output_for_new_study')

# ALLUSION_DICT_PATH = os.path.join(DATA_DIR, 'updated_典故的异性数据.csv')
ALLUSION_DICT_PATH = os.path.join(DATA_DIR, 'variant_allusion_data.csv')

# 标签配置
POSITION_LABELS = {
    'O': 0,
    'B': 1,
    'I': 2
}
