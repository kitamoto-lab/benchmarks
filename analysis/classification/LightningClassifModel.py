import torch.nn as nn
import torch
import torch.optim as optim
from torchvision.models import resnet18, vgg16_bn, vit_b_16
import pytorch_lightning as pl
from torchmetrics import F1Score, ConfusionMatrix, Accuracy


class LightningClassifModel(pl.LightningModule):
    def __init__(self, learning_rate, weights, num_classes, model_name):
        super().__init__()
        self.save_hyperparameters('num_classes')

        # Hyperparams
        self.learning_rate = learning_rate

        # Define Model
        if model_name == "resnet18":
            self.model = resnet18(weights=weights)
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        if model_name == "vgg":
            self.model = vgg16_bn(weights=weights)
            self.model.features[0]= nn.Conv2d(1,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
            self.model.features[-1]=nn.AdaptiveMaxPool2d(7*7)
            self.model.classifier[-1]=nn.Linear(in_features = 4096, out_features=num_classes, bias = True)
        if model_name == "vit":
            self.model = vit_b_16(num_classes=8)
            self.model.conv_proj = nn.Conv2d(in_channels=1, out_channels=768, kernel_size=16, stride=16)

        #loss functions and statistics
        self.loss_fn = nn.CrossEntropyLoss()
        self.compute_micro_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="micro"
        )
        self.compute_macro_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.compute_weighted_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.compute_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        # Collected statistics
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
        self.log_dict({
            "train_loss": loss, 
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        self.log_dict({
            "val_loss": loss, 
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.predicted_labels.append(outputs)
        self.truth_labels.append(labels.int())

        return loss
    
    def on_validation_epoch_end(self):
        all_preds = torch.concat(self.predicted_labels)
        all_truths = torch.concat(self.truth_labels)

        accuracy = self.accuracy(all_preds, all_truths)
        micro_f1 = self.compute_micro_f1(all_preds, all_truths)
        macro_f1 = self.compute_micro_f1(all_preds, all_truths)
        weighted_f1 = self.compute_micro_f1(all_preds, all_truths)
        cm = self.compute_cm(all_preds, all_truths)
        self.log_dict({
            "val_accuracy": accuracy,
            'val_micro_f1': micro_f1,
            'val_macro_f1': macro_f1,
            'val_weighted_f1': weighted_f1,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.logger.experiment.add_embedding(cm, tag=str(self.current_epoch))

        self.predicted_labels.clear()  # free memory
        self.truth_labels.clear()
        print(accuracy)

    def test_step(self, batch, batch_idx):
        loss, outputs, labels = self._common_step(batch)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def _common_step(self, batch):
        images, labels = batch
        labels = labels - 2
        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels.long())
        outputs = torch.argmax(outputs, 1)
        return loss, outputs, labels

    def predict_step(self, batch):
        images, labels = batch
        labels = labels - 2
        outputs = self.forward(images)
        preds = torch.argmax(outputs, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=- 1, verbose=True)
        return [optimizer], [scheduler]