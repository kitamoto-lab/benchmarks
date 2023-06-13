import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms.functional import center_crop

from pathlib import Path
import numpy as np

from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset


class TyphoonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataroot,
        batch_size,
        num_workers,
        split_by="sequence",
        load_data=False,
        dataset_split=(0.8, 0.1, 0.1),
        standardize_range=(170, 300),
        downsample_size=(224, 224),
        cropped=False,
        corruption_ceiling_pct=100,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        data_path = Path(dataroot)
        self.images_path = str(data_path / "image") + "/"
        self.track_path = str(data_path / "metadata") + "/"
        self.metadata_path = str(data_path / "metadata.json")
        self.load_data = load_data
        self.split_by = split_by

        self.dataset_split = dataset_split
        self.standardize_range = standardize_range
        self.downsample_size = downsample_size
        self.cropped = cropped

        self.corruption_ceiling_pct = corruption_ceiling_pct

    def setup(self, stage):
        # Load Dataset
        dataset = DigitalTyphoonDataset(
            str(self.images_path),
            str(self.track_path),
            str(self.metadata_path),
            "grade",
            load_data_into_memory=self.load_data,
            filter_func=self.image_filter,
            transform_func=self.transform_func,
            spectrum="Infrared",
            verbose=False,
        )

        self.train_set, self.val_set, _ = dataset.random_split(
            self.dataset_split, split_by=self.split_by
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def image_filter(self, image):
        return (
            (image.grade() < 7)
            and (image.year() != 2023)
            and (100.0 <= image.long() <= 180.0)
        )  # and (image.mask_1_percent() <  self.corruption_ceiling_pct))

    def transform_func(self, image_batch):
        image_batch = np.clip(
            image_batch, self.standardize_range[0], self.standardize_range[1]
        )
        image_batch = (image_batch - self.standardize_range[0]) / (
            self.standardize_range[1] - self.standardize_range[0]
        )
        if self.downsample_size != (512, 512):
            image_batch = torch.Tensor(image_batch)
            if self.cropped:
                image_batch = center_crop(image_batch, (224, 224))
            else:
                image_batch = torch.reshape(
                    image_batch, [1, 1, image_batch.size()[0], image_batch.size()[1]]
                )
                image_batch = nn.functional.interpolate(
                    image_batch,
                    size=self.downsample_size,
                    mode="bilinear",
                    align_corners=False,
                )
                image_batch = torch.reshape(
                    image_batch, [image_batch.size()[2], image_batch.size()[3]]
                )
            image_batch = image_batch.numpy()
        return image_batch
