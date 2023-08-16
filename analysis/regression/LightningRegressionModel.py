import torch.nn as nn
import torch
import torch.optim as optim
from torchvision.models import resnet18, resnet50, resnet101, vgg16_bn
import pytorch_lightning as pl


class LightningRegressionModel(pl.LightningModule):
    """Resnet Module using lightning architecture"""
    def __init__(self, learning_rate, weights, num_classes, model_name):
        super().__init__()
        self.save_hyperparameters()

        if model_name == "resnet18" : 
            self.model = resnet18(weights=weights)
            self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        if model_name == "resnet50" : 
            self.model = resnet50(weights=weights)
            self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        if model_name == "resnet101" : 
            self.model = resnet101(weights=weights)
            self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        if model_name == "vgg" :
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
        self.logger.experiment.add_embedding(all_couple, tag="couple_label_pred_ep" + str(self.current_epoch))
        
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

        train_loss = torch.mean(torch.tensor(self.all_train_loss))
        train_loss = torch.sqrt(train_loss.clone().detach())

        validation_loss = self.loss_fn(all_preds, all_truths)
        validation_loss = torch.sqrt(validation_loss.clone().detach())
        if train_loss == train_loss: # Check if train_loss != nan
            tensorboard.add_scalars(f"Loss (RMSE)", {'train':train_loss,'validation':validation_loss}, self.current_epoch)

        self.log("validation_loss", validation_loss)

        self.predicted_labels.clear()  # free memory
        self.truth_labels.clear()
        
        print("train_loss:", train_loss.item(), "validation_loss:",  validation_loss.item())

    def test_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        return loss

    def _common_step(self, batch):
        images, labels = batch
        labels = torch.reshape(labels, [labels.size()[0],1])
        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels.float())
        return loss, outputs, labels

    def predict_step(self, batch):
        images, labels = batch
        labels = torch.reshape(labels, [labels.size()[0],1])
        outputs = self.forward(images)
        preds = outputs
        return preds

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45], gamma=0.1, last_epoch=-1, verbose=True)
        return [optimizer] #, [scheduler]