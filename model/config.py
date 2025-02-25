import os

# 模型配置
MODEL_NAME = 'guwenbert-large'
BERT_MODEL_PATH = os.path.join('model', MODEL_NAME)

# 数据配置
MAX_SEQ_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5



# 路径配置
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'   
SAVE_DIR = 'output'
ALLUSION_TYPES_PATH = 'data/updated_典故的异性数据.csv'

# 标签配置
POSITION_LABELS = {
    'O': 0,
    'B': 1,
    'I': 2
}
