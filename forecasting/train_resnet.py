import os
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

from ResNet.hyperparameters import *
from Dataloader.PadSequence import PadSequenceNoTrim, PadSequence
from Dataloader.SequenceDatamodule import TyphoonDataModule

from train_convLSTM import LightningConvLSTM
from evaluation.predict import predict, read_validation_indices


# Flag for if the ConvLSTM output images should be used in training
use_predicted = True  
# use_predicted = False  
loaded_convlstm_model = None
if use_predicted:  
    # Path to ConvLSTM model
    convlstm_checkpoint_path = conv_log_dir + 'lightning_logs/version_12/checkpoints/epoch=0-step=870.ckpt'
    convLSTMmodel = LightningConvLSTM.load_from_checkpoint(convlstm_checkpoint_path)
    convLSTMmodel.eval()
    loaded_convlstm_model = convLSTMmodel


class LightningResnetReg2(pl.LightningModule):
    def __init__(self, learning_rate, weights, use_predicted=False):
        super().__init__()
        self.save_hyperparameters()

        # self.model = resnet50()
        self.model = resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(2, 2), stride=(1, 1), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
        
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        
        self.total_val_loss = 0
        self.use_predicted = use_predicted


    def forward(self, images):
        images = torch.Tensor(images).float()
        output = self.model(images)
        return output

    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        self.log('val_loss', loss)
        self.total_val_loss += loss
        return loss
    
    def on_validation_epoch_end(self):
        tensorboard = self.logger.experiment
        self.log("total_val_loss", self.total_val_loss)
        self.total_val_loss = 0

    def _common_step(self, batch):
        if self.use_predicted:
            images, labels = self.produce_predicted_batch_from_img_batch(batch, loaded_convlstm_model)
        else:
            images, labels = batch
            batch_size, time, channels, height, width = images.size()
            images = torch.reshape(images, (batch_size*time, channels, height, width))
            labels = torch.reshape(labels, [labels.size()[1]])

        labels = torch.reshape(labels, [labels.size()[0],1])
        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels.float())

        return loss, outputs, labels

    def produce_predicted_batch_from_img_batch(self, batch, model):
        images, labels = batch
        batch_size, time, channels, height, width = images.size()
        image_list = torch.reshape(images, (batch_size*time, channels, height, width))
        label_list = torch.reshape(labels, (batch_size, labels.size()[1]))

        with torch.no_grad():
            img_seq = images[:,:prediction_start_hour+1]
            predicted_images = predict(model, img_seq, min(prediction_start_hour, time-(prediction_start_hour+1)))
            expected_labels = labels[:,prediction_start_hour+1:]
            predicted_images = predicted_images.reshape([-1, channels, height, width])
            image_list = torch.concat((image_list, predicted_images), dim=0)
            label_list = torch.concat((label_list, expected_labels), dim=1)

        label_list = torch.reshape(label_list, [label_list.size()[1]])
        return image_list, label_list

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer 

def run_trainer(validation_path, logdir):
    validation_indices = read_validation_indices(validation_path)
    
    if not use_predicted:
        pass_transform = transforms.Compose([PadSequenceNoTrim(max_sequence_length, 
                                                            PAD_token,
                                                            prediction_start_hour,
                                                            prediction_length)])
    else:
        pass_transform = transforms.Compose([PadSequence(max_sequence_length, 
                                                            PAD_token,
                                                            prediction_start_hour,
                                                            prediction_length)])


    data_module = TyphoonDataModule(data_dir,
                                batch_size=1,
                                num_workers=20,
                                split_by=split_by,
                                labels='pressure',
                                load_data=load_data,
                                dataset_split=dataset_split,
                                standardize_range=standardize_range,
                                transform=pass_transform,
                                downsample_size=downsample_size)

    # Use the same train/validation indices
    data_module.setup(0)
    uniq_val = set(validation_indices)
    pre_train_set, pre_val_set = data_module.train_set, data_module.val_set
    new_train_set = []
    for val in pre_train_set.indices:
        if val not in uniq_val:
            new_train_set.append(val)
    for val in pre_val_set.indices:
        if val not in uniq_val:
            new_train_set.append(val)

    data_module.train_set = Subset(data_module.dataset, new_train_set)
    data_module.val_set = Subset(data_module.dataset, validation_indices)

    resnet_model = LightningResnetReg2(learning_rate=0.00001, weights=None, use_predicted=use_predicted)
    checkpoint_callback = ModelCheckpoint(monitor='total_val_loss', mode='min', every_n_epochs=1, save_top_k=-1)
    trainer = Trainer(max_epochs=35,
                      accelerator=accelerator,
                      default_root_dir=logdir, 
                      callbacks=[checkpoint_callback],
                      num_sanity_val_steps=0)
    
    # trainer.validate(model=resnet_model, dataloaders=data_module)

    trainer.fit(resnet_model, data_module)


if __name__ == '__main__':
    # Path to validation set dataset indices
    validation_path = conv_log_dir + 'lightning_logs/version_12/validation_indices.txt'
    
    run_trainer(validation_path, log_dir)

