import torch

# Training Hyperparameters
LEARNING_RATE     = 0.0001
BATCH_SIZE        = 16
NUM_WORKERS       = 24
MAX_EPOCHS        = 50


# DATASET
WEIGHTS           = None
LABELS            = 'wind'
SPLIT_BY          = 'sequence'
LOAD_DATA         = 'all_data'
DATASET_SPLIT     = (0.8, 0.2, 0)
STANDARDIZE_RANGE = (170, 300)
DOWNSAMPLE_SIZE   = (224, 224)
NUM_CLASSES       = 1

# Computation
ACCELERATOR       = 'gpu' if torch.cuda.is_available() else 'cpu'
DEVICES           = [1]
DATA_DIR          = '/dataset/typhoon/WP/'
LOG_DIR           = "/app/digtyp/FrameClassification/Comparison/tb_logs"