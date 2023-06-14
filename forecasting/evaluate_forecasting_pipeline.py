# import libraries
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import Dataset, Subset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from ResNet.hyperparameters import *
from Dataloader.PadSequence import PadSequence
from Dataloader.SequenceDatamodule import TyphoonDataModule

from train_convLSTM import LightningConvLSTM
from train_resnet import LightningResnetReg2
from evaluation.predict import predict, read_validation_indices


def compute_stats(predicted_labels, expected_labels, save_location):
    # pred_label_size: [12], exp label size: [12]
    diff = torch.abs(expected_labels - predicted_labels)
    pct_diff = diff / expected_labels
    se = torch.square(diff)

    with open(save_location, 'w') as f:
        savestring = str(predicted_labels.tolist()) + '\n' + str(expected_labels.tolist()) + '\n' 
        savestring += str(diff.tolist()) + '\n' + str(pct_diff.tolist()) + '\n' + str(se.tolist())
        f.write(savestring)

    return diff, pct_diff, se

def plot_forecast_plot(input_labels, predicted_labels, expected_labels, save_location):
    input_labels, predicted_labels, expected_labels = input_labels.detach(), predicted_labels.detach(), expected_labels.detach()
    input_x = list(range(len(input_labels)+len(expected_labels)))
    forecasted_x = list(range(12, 12+len(predicted_labels)))

    total_ground_truth = torch.cat((input_labels, expected_labels))
    plt.plot(input_x, total_ground_truth.cpu(), 'k', label='Ground truth')
    plt.plot(forecasted_x, predicted_labels.cpu(), 'r:', label='Forecast')
    plt.ylim(900, 1020)

    plt.xlabel("Hour")
    plt.ylabel("Pressure (hPa)")
    plt.title("Forecasted Pressure by Hour from Typhoon Start")
    plt.legend()
    plt.savefig(save_location)
    plt.close()



def run_evaluation(validation_path, convLSTMpath, resnetPath, logdir, startbatch):

    validation_indices = read_validation_indices(validation_path)
    gpu_device = torch.device('cuda')
    cpu_device = torch.device('cpu')

    data_module = TyphoonDataModule(data_dir,
                                batch_size=1,
                                num_workers=0,
                                split_by=split_by,
                                labels='pressure',
                                load_data=load_data,
                                dataset_split=dataset_split,
                                standardize_range=standardize_range,
                                transform=transforms.Compose([
                                            PadSequence(max_sequence_length, 
                                                        PAD_token,
                                                        prediction_start_hour,
                                                        prediction_length),
                                ]),
                                downsample_size=downsample_size)

    data_module.setup(0)
    data_module.val_set = Subset(data_module.dataset, validation_indices)

    torch.set_grad_enabled(False)

    convLSTMmodel = LightningConvLSTM.load_from_checkpoint(convLSTMpath)
    convLSTMmodel.eval()

    resnetModel = LightningResnetReg2.load_from_checkpoint(resnetPath)
    resnetModel.use_predicted = False
    resnetModel.eval()

    val_loader = data_module.val_dataloader()

    num_instances = 0
    total_diffs = torch.zeros([12]).to(resnetModel.device)
    total_pct_diffs = torch.zeros([12]).to(resnetModel.device)
    total_ses = torch.zeros([12]).to(resnetModel.device)
    for i, batch in enumerate(tqdm(val_loader)):
        if i < startbatch:
            continue

        images, labels = batch
        batch_size, time, channels, height, width = images.size()
        images, labels = images.to(convLSTMmodel.device), labels.to(resnetModel.device)
        img_seq = images[:,0:12]

        predicted_images = predict(convLSTMmodel, img_seq, 12)

        expected_labels = labels[:,12:].reshape([-1])
        pred_batch_size, pred_num_frames_to_predict, pred_channels, pred_height, pred_width = predicted_images.size()
        predicted_images = predicted_images.reshape([-1, pred_channels, pred_height, pred_width])
        
        predicted_labels = resnetModel.forward(predicted_images)
        predicted_labels = predicted_labels.reshape([pred_num_frames_to_predict*pred_batch_size])

        diff, pct_diff, se = compute_stats(predicted_labels, expected_labels, logdir + f'plot_{i}.txt')

        total_diffs += diff
        total_pct_diffs += pct_diff
        total_ses += se
        num_instances += 1

        plot_forecast_plot(labels[:,:12].reshape([-1]), predicted_labels, expected_labels, logdir + f'plot_{i}.png')


    total_diffs = total_diffs / num_instances
    total_pct_diffs = total_pct_diffs / num_instances
    total_ses = total_ses / num_instances
    total_ses = torch.sqrt(total_ses)
    print('Diff: ', total_diffs.tolist())
    print('PctDiff: ', total_pct_diffs.tolist())
    print('RMSE: ', total_ses.tolist())
    save_obj = {'diff':total_diffs,
                'pctdiff':total_pct_diffs,
                'rmse':total_ses}
    torch.save(save_obj, logdir + 'collected_stats.pt')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--startbatch', default=0, type=int)
    args = parser.parse_args()

    # Path to validation set dataset indices
    validation_path = conv_log_dir + 'lightning_logs/version_8/validation_indices.txt'
    
    # Path to ConvLSTM model
    convlstm_checkpoint_path = conv_log_dir + 'lightning_logs/version_8/checkpoints/epoch=244-step=114450.ckpt'
    
    # Path to ResNet regression model
    resnet_checkpoint_path = log_dir + 'lightning_logs/version_14/checkpoints/epoch=34-step=38115.ckpt'

    logdir = str(Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'Pipeline_logs' / 'forecast_plots') + '/'    
    run_evaluation(validation_path, convlstm_checkpoint_path, resnet_checkpoint_path, logdir, args.startbatch)

