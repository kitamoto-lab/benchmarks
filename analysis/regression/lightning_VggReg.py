import torch.nn as nn
import torch
import torch.optim as optim
from torchvision.models import vgg16_bn
import pytorch_lightning as pl


class LightningVggReg(pl.LightningModule):
    """Resnet Module using lightning architecture"""
    def __init__(self, learning_rate, weights, num_classes):
        super().__init__()
        self.save_hyperparameters()

        self.model = vgg16_bn(num_classes=num_classes, weights=weights)
        self.model.features[0]= nn.Conv2d(1,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.model.features[-1]=nn.AdaptiveMaxPool2d(7*7)
        self.model.classifier[-1]=nn.Linear(in_features = 4096, out_features=1, bias = True)
        
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        
        self.all_train_loss = []

        self.truth_labels = []
        self.predicted_labels = []

    def forward(self, images):
        images = torch.Tensor(images).float()
        images = torch.reshape(
            images, [images.size()[0], 1, images.size()[1], images.size()[2]]
        )
        output = self.model(images)
        return output

    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        self.all_train_loss.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        self.predicted_labels.append(outputs)
        self.truth_labels.append(labels.float())
        return loss
    
    def on_validation_epoch_end(self):
        """Save logs of every epochs : couple (truth, predictions) and validation loss"""
        tensorboard = self.logger.experiment

        all_preds = torch.concat(self.predicted_labels)
        all_truths = torch.concat(self.truth_labels)
        all_couple = torch.cat((all_truths, all_preds), dim=1)
        self.logger.experiment.add_embedding(all_couple, tag="couple_label_pred_ep" + str(self.current_epoch) + ".tsv")
        
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
        
        train_loss = torch.mean(torch.tensor(self.all_train_loss))
        train_loss = torch.sqrt(torch.tensor(train_loss))

        validation_loss = self.loss_fn(all_preds, all_truths)
        validation_loss = torch.sqrt(torch.tensor(validation_loss))
        tensorboard.add_scalars(f"Loss (RMSE)", {'train':train_loss,'validation':validation_loss},self.current_epoch)

        self.log("validation_loss", validation_loss)

        self.predicted_labels.clear()  # free memory
        self.truth_labels.clear()

    def test_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
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
