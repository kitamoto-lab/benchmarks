import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from FrameDatamodule import TyphoonDataModule
from LightningRegressionModel import LightningRegressionModel
from pytorch_lightning.callbacks import ModelCheckpoint

import config
from argparse import ArgumentParser

from datetime import datetime
import torch
from torchvision.transforms.functional import center_crop

from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
from pathlib import Path
import numpy as np

start_time_str = str(datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))

class TripleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indice_list, start, end):
        super(TripleDataset).__init__()
        self.dataset = dataset
        self.start = start
        self.end = end
        self.indice_list = indice_list

    def __getitem__(self, idx):
        original_idx = self.indice_list[self.start + idx]
        image_0, labels_0 = self.dataset[original_idx]
        image_moins1, labels_moins1 = self.dataset[original_idx-1]
        image_moins2, labels_moins2 = self.dataset[original_idx-2]
        image_x3 = np.array([image_moins2, image_moins1, image_0])
        image_tensor = torch.Tensor(image_x3)
        image_tensor = center_crop(image_tensor, (224, 224))
        image_x3 = image_tensor.numpy()
        return image_x3, float(labels_0[8])
    
    def __len__(self):
        return (self.end - self.start)

def custom_parse_args(args):
    """Argument parser, verify if model_name, device, label, size and cropped arguments are correctly initialized"""

    args_parsing = ""
    if args.model_name not in ["resnet18", "resnet50","resnet101", "vgg"]:
        args_parsing += "Please give model_name among resnet18, 50, 101 or vgg\n"
    if args.size not in ["512", "224", 512, 224]:
        args_parsing += "Please give size equals to 512 or 224\n"
    if args.cropped not in ["False", "True", "false", "true", False, True]:
        args_parsing += "Please give cropped equals to False or True\n"
    if int(args.device) not in range(torch.cuda.device_count()):
        args_parsing += "Please give a device number in the range (0, %d)\n" %torch.cuda.device_count()
    if args.labels not in ["wind", "pressure"]:
        args_parsing += "Please give size equals to wind or pressure\n"

    if args_parsing != "": 
        print(args_parsing)
        raise ValueError("Some arguments are not initialized correctly")
    
    if args.size == '512' or args.size == 512:
        args.size = (512,512)
    elif args.size == '224'  or args.size == 224:
        args.size = (224, 224)
    
    if args.cropped in ["False", "false"]: args.cropped = False
    if args.cropped == ["True", "true"]: args.cropped = True

    if args.device == None:
        args.device = config.DEVICES
    else:
        args.device = [int(args.device)]

    return args

def train(hparam):
    """Launch a training with the lightning library and the arguments given in the python command and the hyper parameters in the config file"""
    hparam = custom_parse_args(hparam)

    logger_name = hparam.labels + "_" + hparam.model_name + "_" + str(hparam.size[0])
    if hparam.cropped: logger_name += "_cropped"
    else : logger_name += "_no-crop"
    logger_name += "_3img_method"

    logger = TensorBoardLogger(
        save_dir="results",
        name= logger_name,
        default_hp_metric=False,
    )

    # Log all hyper parameters
    logger.log_hyperparams({
        'start_time': start_time_str,
        'LEARNING_RATE': config.LEARNING_RATE,
        'BATCH_SIZE': config.BATCH_SIZE,
        'NUM_WORKERS': config.NUM_WORKERS,
        'MAX_EPOCHS': config.MAX_EPOCHS,
        'WEIGHTS': config.WEIGHTS, 
        'LABEL' : hparam.labels,
        'SPLIT_BY': config.SPLIT_BY, 
        'LOAD_DATA': config.LOAD_DATA, 
        'DATASET_SPLIT': config.DATASET_SPLIT, 
        'STANDARDIZE_RANGE': config.STANDARDIZE_RANGE, 
        'DOWNSAMPLE_SIZE': hparam.size, 
        'CROPPED': hparam.cropped,
        'NUM_CLASSES': config.NUM_CLASSES, 
        'ACCELERATOR': config.ACCELERATOR, 
        'DEVICES': hparam.device, 
        'DATA_DIR': config.DATA_DIR, 
        'MODEL_NAME': hparam.model_name,
        "COMMENT": "3images at t-2, t-1 and t0. Use of a Conv3d for the first layer",
        # "Scheduler": "Start from 0.001 and divide by 10 every 15epochs"
        })

    # # Set up dataset
    data_module = TyphoonDataModule(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        labels=hparam.labels,
        split_by=config.SPLIT_BY,
        load_data=config.LOAD_DATA,
        dataset_split=config.DATASET_SPLIT,
        standardize_range=config.STANDARDIZE_RANGE,
        downsample_size=hparam.size,
        cropped=hparam.cropped
    )

    # model selection
    regression_model = LightningRegressionModel(
        learning_rate=config.LEARNING_RATE,
        weights=config.WEIGHTS,
        num_classes=config.NUM_CLASSES,
        model_name = hparam.model_name
    )

    # Callback for model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath= logger.save_dir + '/' + logger.name + '/version_%d/checkpoints/' % logger.version,
        filename='model_{epoch}',
        monitor='validation_loss', 
        verbose=True,
        every_n_epochs=1,
        save_top_k = 5
        )

    # Setting up the lightning trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=hparam.device,
        max_epochs=config.MAX_EPOCHS,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback]
    )

    # Launch training session
    DATA_DIR = Path('/dataset/typhoon/WP/')

    images_path = str(DATA_DIR / "image")  + "/"
    track_path = str(DATA_DIR / "metadata")  + "/"
    metadata_path = str(DATA_DIR / "metadata.json")
    label_list = ["year", "month", "day", "hour", "grade", "lat", "lng", "pressure", "wind", "dir50", "long50", "short50", "dir30", "long30", "short30", "landfall", 'interpolated', 'filename', 'mask_1', 'mask_1_percent']
    n_labels = len(label_list)

    def import_dataset():
        print('importing dataset...')
        dataset = DigitalTyphoonDataset(
            str(images_path),
            str(track_path),
            str(metadata_path),
            label_list,
            load_data_into_memory="track",
            filter_func=image_filter,
            transform_func=None,
            spectrum="Infrared",
            verbose=False,
        )
        return dataset
    
    def image_filter(image):
        return (
            (image.grade() < 6)
            and (image.grade() > 2)
            and (image.year() < 2023)
            and (100.0 <= image.long() <= 180.0)
        )  # and (image.mask_1_percent() <  self.corruption_ceiling_pct))

    dataset = import_dataset()

    indice_list = np.load("dataset2_indice_list.npy")
    train_set = TripleDataset(dataset, indice_list, 2, 83825)
    val_set = TripleDataset(dataset, indice_list, 83825, len(indice_list))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    trainer.fit(regression_model, train_loader, val_loader)
    
    return "training finished"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", default='resnet18')
    parser.add_argument("--size", default=config.DOWNSAMPLE_SIZE)
    parser.add_argument("--cropped", default=True) # if size is 512, cropped argument doesn't have any impact
    parser.add_argument("--device", default=config.DEVICES)
    parser.add_argument("--labels", default=config.LABELS)
    args = parser.parse_args()

    print(train(args))
