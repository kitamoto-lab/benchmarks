import numpy as np
import torch

# Hyperparameters
learning_rate     = 0.0001
batch_size        = 1
num_workers       = 4
max_epochs        = 300
weights           = None
split_by          = 'sequence'
load_data         = False
dataset_split     = (0.8, 0.2, 0.0)
standardize_range = (170, 300)
downsample_size   = (128,128)
PAD_token         = 4
accelerator       = 'gpu' if torch.cuda.is_available() else 'cpu'
hidden_dim        =  128


data_dir          = '/data/'
log_dir = str(Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'ResNet_logs') + '/'

max_sequence_length = 528

prediction_start_hour = 12
prediction_length = 12

num_layers = 3
