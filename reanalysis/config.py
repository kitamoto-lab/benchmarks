import torch

# Training Hyperparameters
LEARNING_RATE     = 0.0001
BATCH_SIZE        = 16
NUM_WORKERS       = 16
MAX_EPOCHS        = 101
NB_RUNS           = 5
TESTING_VERSION   = (0,1,2,3,4)



# DATASET
WEIGHTS           = None
LABELS            = 'pressure'
SPLIT_BY          = 'sequence'
LOAD_DATA         = 'all_data'
DATASET_SPLIT     = (0.8, 0.1, 0.1)
STANDARDIZE_RANGE = (170, 350)
DOWNSAMPLE_SIZE   = (224, 224)
NUM_CLASSES       = 1
TYPE_SAVE         = 'standard' #'standard' or 'same_size'

# Computation
ACCELERATOR       = 'gpu' if torch.cuda.is_available() else 'cpu'
DEVICE            = [0]
DATA_DIR          = '/app/datasets/wnp/'
LOG_DIR           = "/app/pyphoon2/reanalysis/tb_logs"