import torch

# Training Hyperparameters
LEARNING_RATE     = 0.0005
BATCH_SIZE        = 16
NUM_WORKERS       = 24
MAX_EPOCHS        = 150

# Dataset parameters
WEIGHTS           = "DEFAULT"
LABELS            = 'wind' # Overwritten if labels argument is given
SPLIT_BY          = 'sequence'
LOAD_DATA         = 'all_data'
DATASET_SPLIT     = (0.8, 0.2, 0)
STANDARDIZE_RANGE = (170, 300)
DOWNSAMPLE_SIZE   = 224 # Overwritten if size argument is given
NUM_CLASSES       = 1

# Computation parameters
ACCELERATOR       = 'gpu' if torch.cuda.is_available() else 'cpu'
DEVICES           = 0 # Overwritten if device argument is given
DATA_DIR          = '/dataset/'