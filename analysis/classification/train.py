import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from FrameDatamodule import TyphoonDataModule
from LightningClassifModel import LightningClassifModel
from pytorch_lightning.callbacks import ModelCheckpoint

import config
from argparse import ArgumentParser

from datetime import datetime
import torch

start_time_str = str(datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))

def custom_parse_args(args):
    """Argument parser, verify if model_name, device, label, size and cropped arguments are correctly initialized"""

    args_parsing = ""
    if args.model_name not in ["resnet18", "vgg", "vit"]:
        args_parsing += "Please give model_name among resnet18, vgg or vit\n"
    if args.size not in ["512", "224", 512, 224]:
        args_parsing += "Please give size equals to 512 or 224\n"
    if args.cropped not in ["False", "True", "false", "true", False, True]:
        args_parsing += "Please give cropped equals to False or True\n"
    if torch.cuda.is_available() and int(args.device) not in range(torch.cuda.device_count()):
        args_parsing += "Please give a device number in the range (0, %d)\n" %torch.cuda.device_count()

    if args_parsing != "": 
        print(args_parsing)
        raise ValueError("Some arguments are not initialized correctly")
    
    if args.size == '512' or args.size == 512:
        args.size = (512,512)
    elif args.size == '224'  or args.size == 224:
        args.size = (224, 224)
    
    if args.cropped in ["False", "false"]: args.cropped = False
    if args.cropped in ["True", "true"]: args.cropped = True

    if args.device == None:
        args.device = config.DEVICES
    else:
        args.device = [int(args.device)]

    return args

def train(hparam):
    """Launch a training with the PytorchLightning workflow, the arguments given in the python command and the hyper parameters in the config file"""
    hparam = custom_parse_args(hparam)

    logger_name = hparam.model_name + "_" + str(hparam.size[0])
    if hparam.cropped: logger_name += "_cropped"
    else : logger_name += "_no-crop"

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
        'LABEL' : "grade",
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
        })

    # Set up dataset
    data_module = TyphoonDataModule(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        split_by=config.SPLIT_BY,
        load_data=config.LOAD_DATA,
        dataset_split=config.DATASET_SPLIT,
        standardize_range=config.STANDARDIZE_RANGE,
        downsample_size=hparam.size,
        cropped=hparam.cropped
    )

    # Train
    classification_model = LightningClassifModel(
        learning_rate=config.LEARNING_RATE,
        weights=config.WEIGHTS,
        num_classes=config.NUM_CLASSES,
        model_name=hparam.model_name
    )

    # Callback for model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath= logger.save_dir + '/' + logger.name + '/version_%d/checkpoints/' % logger.version,
        filename='model_{epoch}',
        monitor='val_loss', 
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
    trainer.fit(classification_model, data_module)

    return "training finished"

# Main execution block
if __name__ == "__main__":
    # Parse command-line arguments using argparse
    parser = ArgumentParser()
    parser.add_argument("--model_name", default='vgg')
    parser.add_argument("--size", default=config.DOWNSAMPLE_SIZE)
    parser.add_argument("--cropped", default=True) # if size is 512, cropped argument doesn't have any impact
    parser.add_argument("--device", default=config.DEVICES)
    args = parser.parse_args()

    print(train(args))
