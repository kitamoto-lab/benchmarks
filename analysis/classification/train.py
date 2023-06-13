import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from FrameDatamodule import TyphoonDataModule
from lightning_resnet import LightningResnet
from lightning_vgg import LightningVGG
from lightning_vit import LightningVit
from pytorch_lightning.callbacks import ModelCheckpoint

import config
from argparse import ArgumentParser

from datetime import datetime

start_time_str = str(datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))

def main(hparam):    
    logger_name = hparam.model_name + "_" + hparam.size
    if hparam.cropped: logger_name += "_cropped"

    if args.size == None:
        size = config.DOWNSAMPLE_SIZE
    elif args.size == '512':
        size = (512,512)
    elif args.size == '224':
        size = (224, 224)

    logger = TensorBoardLogger(
        save_dir="tb_logs",
        name= logger_name,
        default_hp_metric=False,
    )

    logger.log_hyperparams({
        'start_time': start_time_str,
        'LEARNING_RATE': config.LEARNING_RATE,
        'BATCH_SIZE': config.BATCH_SIZE,
        'NUM_WORKERS': config.NUM_WORKERS,
        'MAX_EPOCHS': config.MAX_EPOCHS,
        'WEIGHTS': config.WEIGHTS, 
        'LABEL' : "wind",
        'SPLIT_BY': config.SPLIT_BY, 
        'LOAD_DATA': config.LOAD_DATA, 
        'DATASET_SPLIT': config.DATASET_SPLIT, 
        'STANDARDIZE_RANGE': config.STANDARDIZE_RANGE, 
        'DOWNSAMPLE_SIZE': size, 
        'CROPPED': hparam.cropped,
        'NUM_CLASSES': config.NUM_CLASSES, 
        'ACCELERATOR': config.ACCELERATOR, 
        'DEVICES': hparam.device, 
        'DATA_DIR': config.DATA_DIR, 
        'LOG_DIR': config.LOG_DIR,
        'MODEL_NAME': hparam.model_name,
        })

    # Set up data
    data_module = TyphoonDataModule(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        split_by=config.SPLIT_BY,
        load_data=config.LOAD_DATA,
        dataset_split=config.DATASET_SPLIT,
        standardize_range=config.STANDARDIZE_RANGE,
        downsample_size=size,
        cropped=hparam.cropped
    )

    # Train
    resnet = LightningResnet(
        learning_rate=config.LEARNING_RATE,
        weights=config.WEIGHTS,
        num_classes=config.NUM_CLASSES,
    )
    vgg = LightningVGG(
        learning_rate=config.LEARNING_RATE,
        weights=config.WEIGHTS,
        num_classes=config.NUM_CLASSES,
    )
    vit = LightningVit(
        learning_rate=config.LEARNING_RATE,
        weights=config.WEIGHTS,
        num_classes=config.NUM_CLASSES,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath= logger.save_dir + '/' + logger.name + '/version_%d/checkpoints/' % logger.version,
        filename='model_{epoch}',
        monitor='val_loss', 
        verbose=True,
        every_n_epochs=1,
        save_top_k = 5
        )

    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=hparam.device,
        max_epochs=config.MAX_EPOCHS,
        default_root_dir=config.LOG_DIR,
        callbacks=[checkpoint_callback]
    )

    if hparam.model_name[:6]=='resnet': trainer.fit(resnet, data_module)
    if hparam.model_name=='vgg':trainer.fit(vgg, data_module)
    if hparam.model_name=='vit':trainer.fit(vit, data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", default='vgg')
    parser.add_argument("--size")
    parser.add_argument("--cropped", default=False)
    parser.add_argument("--device")
    args = parser.parse_args()

    if args.device == None:
        args.device = config.DEVICES
    else:
        args.device = [int(args.device)]
    
    main(args)

