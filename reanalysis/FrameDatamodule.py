import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pathlib import Path
import numpy as np

from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset


class TyphoonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataroot,
        batch_size,
        num_workers,
        labels = 'grade',
        split_by="sequence",
        load_data=False,
        dataset_split=(0.8, 0.1, 0.1),
        standardize_range=(150, 350),
        downsample_size=(224, 224),
        corruption_ceiling_pct=100,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        data_path = Path(dataroot)
        self.images_path = str(data_path / "image") + "/"
        self.track_path = str(data_path / "track") + "/"
        self.metadata_path = str(data_path / "metadata.json")
        self.load_data = load_data
        self.split_by = split_by
        self.labels = labels

        self.dataset_split = dataset_split
        self.standardize_range = standardize_range
        self.downsample_size = downsample_size

        self.corruption_ceiling_pct = corruption_ceiling_pct

    def setup(self, stage):
        # Load Dataset
        dataset = DigitalTyphoonDataset(
            str(self.images_path),
            str(self.track_path),
            str(self.metadata_path),
            self.labels,
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
            (image.grade() < 6)
            and (image.grade() > 2)
            and (image.interpolated() == False)
            and (image.year() != 2023)
            and (100.0 <= image.long() <= 180.0)
        )  # and (image.mask_1_percent() <  self.corruption_ceiling_pct))

    def transform_func(self, image_ray):
        image_ray = np.clip(
            image_ray, self.standardize_range[0], self.standardize_range[1]
        )
        image_ray = (image_ray - self.standardize_range[0]) / (
            self.standardize_range[1] - self.standardize_range[0]
        )
        if self.downsample_size != (512, 512):
            image_ray = torch.Tensor(image_ray)
            image_ray = torch.reshape(
                image_ray, [1, 1, image_ray.size()[0], image_ray.size()[1]]
            )
            image_ray = nn.functional.interpolate(
                image_ray,
                size=self.downsample_size,
                mode="bilinear",
                align_corners=False,
            )
            image_ray = torch.reshape(
                image_ray, [image_ray.size()[2], image_ray.size()[3]]
            )
            image_ray = image_ray.numpy()
        return image_ray
