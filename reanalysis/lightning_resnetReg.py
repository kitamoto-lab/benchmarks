import torch.nn as nn
import torch
import torch.optim as optim
from torchvision.models import resnet18
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError



class LightningResnetReg(pl.LightningModule):
    def __init__(self, learning_rate, weights, num_classes):
        super().__init__()
        self.save_hyperparameters()

        self.model = resnet18(num_classes=1, weights=weights)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
        
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        self.accuracy = MeanSquaredError(squared = False)
        self.compt = 1
        self.predicted_labels = []
        self.truth_labels = []


    def forward(self, images):
        images = torch.Tensor(images).float()
        images = torch.reshape(
            images, [images.size()[0], 1, images.size()[1], images.size()[2]]
        )
        output = self.model(images)
        return output

    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        accuracy = self.accuracy(outputs, labels)
        self.log_dict({
            "train_loss": loss, 
            "train_RMSE": accuracy
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        self.log("validation_loss", loss,
            on_step=False, on_epoch=True, sync_dist=True)
        self.predicted_labels.append(outputs)
        self.truth_labels.append(labels.float())
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        self.log("test_loss", loss,
            on_step=False, on_epoch=True, sync_dist=True)
        self.predicted_labels.append(outputs)
        self.truth_labels.append(labels.float())
        return loss

    def _common_step(self, batch):
        images, labels = batch
        labels = labels - 2
        labels = torch.reshape(labels, [labels.size()[0],1])
        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels.float())
        return loss, outputs, labels

    def predict_step(self, batch):
        images, labels = batch
        labels = labels - 2
        labels = torch.reshape(labels, [labels.size()[0],1])
        outputs = self.forward(images)
        preds = outputs
        return preds

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.learning_rate)
    
    def on_validation_epoch_end(self):
        
        tensorboard = self.logger.experiment
        all_preds = torch.concat(self.predicted_labels)
        all_truths = torch.concat(self.truth_labels)
        all_couple = torch.cat((all_truths, all_preds), dim=1)
        wind_values = torch.unique(all_truths)
        pred_means = []
        pred_std = []
        pred_n = []
        for value in wind_values:
            # find all the couple (truth, preds) where truth == value and compute the mean of all the prediction for this value
            m = torch.mean((all_couple[torch.where(all_couple[:,0] == value)][:,1].float()))
            std = torch.std((all_couple[torch.where(all_couple[:,0] == value)][:,1].float()))
            n = len(all_couple[torch.where(all_couple[:,0] == value)][:,1].float())
            pred_means.append(m)
            pred_std.append(std)
            pred_n.append(n)

        # Log regression line graph every 5 epochs
        if(self.current_epoch %5 == 0 ):            
            for i in range(len(wind_values)):
                tensorboard.add_scalars(f"epoch_{self.current_epoch}",{'pred_mean':pred_means[i],'truth':wind_values[i]},wind_values[i])
                tensorboard.add_scalars(f"epoch_{self.current_epoch}_stats",{'pred_std':pred_std[i],'pred_n':pred_n[i]},wind_values[i])
        
        
        self.log("validation_RMSE", self.accuracy(all_preds,all_truths),
            on_step=False, on_epoch=True, sync_dist=True)
        self.predicted_labels.clear()  # free memory
        self.truth_labels.clear()
        
    def on_test_epoch_end(self):
        tensorboard= self.logger.experiment
        
        all_preds = torch.concat(self.predicted_labels)
        all_truths = torch.concat(self.truth_labels)
        all_couple = torch.cat((all_truths, all_preds), dim=1)        
        self.logger.experiment.add_embedding(all_couple, tag="couple_label_pred_ep" + str(self.compt) + ".tsv")
        unique_values = torch.unique(all_truths)
        pred_means = []
        pred_std = []
        pred_n = []
        for value in unique_values:
            # find all the couple (truth, preds) where truth == value and compute the mean of all the prediction for this value
            m = torch.mean((all_couple[torch.where(all_couple[:,0] == value)][:,1].float()))
            std = torch.std((all_couple[torch.where(all_couple[:,0] == value)][:,1].float()))
            n = len(all_couple[torch.where(all_couple[:,0] == value)][:,1].float())
            pred_means.append(m)
            pred_std.append(std)
            pred_n.append(n)

        # Log regression line graph every 5 epochs
        if(self.current_epoch %5 == 0 ):            
            for i in range(len(unique_values)):
                tensorboard.add_scalars(f"test_{self.compt}",{'pred_mean':pred_means[i],'truth':unique_values[i]},unique_values[i])
                tensorboard.add_scalars(f"test_{self.compt}_stats",{'pred_std':pred_std[i],'pred_n':pred_n[i]},unique_values[i])
        
        Accuracy = self.accuracy(all_preds,all_truths)
        self.log(f"test_{self.compt}_RMSE", Accuracy,
            on_step=False, on_epoch=True, sync_dist=True)
        with open("log.txt","a+") as file:
            file.write(f"test_{self.compt}_RMSE : {Accuracy} \n")
        self.predicted_labels.clear()  # free memory
        self.truth_labels.clear()
        self.compt +=1