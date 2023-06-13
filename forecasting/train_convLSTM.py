# import libraries
import os
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import resnet18
from torchmetrics import F1Score, ConfusionMatrix, Accuracy
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from ConvLSTM.hyperparameters import *
from Dataloader.PadSequence import PadSequence
from Dataloader.SequenceDatamodule import TyphoonDataModule

from ConvLSTM.Seq2Seq import Seq2Seq
from evaluation.predict import read_validation_indices


class LightningConvLSTM(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Hyperparams
        self.learning_rate = learning_rate

        # Define Model

        # input size = (batch, channels, time, width, height)
        self.model = Seq2Seq(num_channels=1, num_kernels=hidden_dim, 
                            kernel_size=(3, 3), padding=(1, 1), activation="relu", 
                            frame_size=downsample_size, num_layers=num_layers)

        self.criterion = nn.MSELoss()

        self.total_val_loss = 0

    def forward(self, images):
        images = images.permute([0, 2, 1, 3, 4])
        out = self.model(images)
        return out

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, train=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log('val_loss', loss)
        self.total_val_loss += loss.detach().item()
        return loss
    
    def on_validation_epoch_end(self):
        self.log('total_val_loss', self.total_val_loss, sync_dist=True)
        self.total_val_loss = 0

        if self.current_epoch == 0:
            version = self.logger.version
            val_set = self.trainer.datamodule.val_set
            with open(str(Path(log_dir) / 'lightning_logs' / f'version_{version}' / f'validation_indices.txt'), 'w') as f:
                f.write(str([int(idx) for idx in val_set.indices]))

        return

    def loss_fn(self, predicted, target):
        loss = self.criterion(predicted, target)
        return loss

    def create_video(self, x, y_hat, y):        
        # predictions with input for illustration purposes
        preds = torch.cat([x.cpu(), y_hat.unsqueeze(2).cpu()], dim=1)[0]

        # entire input and ground truth
        y_plot = torch.cat([x.cpu(), y.unsqueeze(2).cpu()], dim=1)[0]

        # error (l2 norm) plot between pred and ground truth
        difference = (torch.pow(y_hat[0] - y[0], 2)).detach().cpu()
        zeros = torch.zeros(difference.shape)
        difference_plot = torch.cat([zeros.cpu().unsqueeze(0), difference.unsqueeze(0).cpu()], dim=1)[
            0].unsqueeze(1)

        # concat all images
        final_image = torch.cat([preds, y_plot, difference_plot], dim=0)

        num_prev = prediction_start_hour
        num_ahead = 1
        # make them into a single grid image file
        grid = torchvision.utils.make_grid(final_image, nrow=num_prev+num_ahead)

        return grid

    def produce_batch_from_img_batch(self, batch):
        images, labels = batch
        batch_size, time, channels, height, width = images.size()
        newimages = torch.Tensor([]).to(self.device)
        i = 0
        while i+prediction_start_hour+1 < time:
            newimages = torch.concat((newimages, images[:,i:i+prediction_start_hour+1]),dim=0)
            i += 1
        return newimages, labels

    def _common_step(self, batch, train=False):
        images, labels = self.produce_batch_from_img_batch(batch)

        batch_size, time, channels, height, width = images.size()
        model_input = images[:,:prediction_start_hour]
        expected = images[:,prediction_start_hour]

        outputs = self.forward(model_input)
        loss = self.loss_fn(outputs, expected)

        if train:
            if self.global_step % 250 == 0:
                final_image = self.create_video(model_input, outputs, expected)

                self.logger.experiment.add_image(
                    'epoch_' + str(self.current_epoch) + '_step' + str(self.global_step) + '_generated_images',
                    final_image, 0)
                plt.close()

        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(opt.beta_1, opt.beta_2))
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=0.000001, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, eta_min=0.00001)
        return [optimizer], [scheduler]


def run_trainer():
    data_module = TyphoonDataModule(data_dir,
                                batch_size=batch_size,
                                num_workers=num_workers,
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

    model = LightningConvLSTM()

    checkpoint_callback = ModelCheckpoint(monitor='total_val_loss', mode='min', every_n_epochs=5, save_top_k=-1)

    trainer = Trainer(max_epochs=max_epochs,
                      accelerator=accelerator,
                      default_root_dir=log_dir, 
                      callbacks=[checkpoint_callback])

    trainer.fit(model, data_module, ckpt_path=convlstm_checkpoint_path)


if __name__ == '__main__':
    run_trainer()

