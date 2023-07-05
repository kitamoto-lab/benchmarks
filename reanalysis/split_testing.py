import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_resnetReg import LightningResnetReg
import config
import loading
import torch
from torch import nn
import os

from pathlib import Path
import numpy as np

from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset

def main():
    logger_old = TensorBoardLogger("tb_logs", name="resnet_test_old_same")
    logger_recent = TensorBoardLogger("tb_logs", name="resnet_test_recent_same")
    logger_now = TensorBoardLogger("tb_logs", name="resnet_test_now_same")

    # Set up data
    data_root = config.DATA_DIR
    batch_size=config.BATCH_SIZE
    num_workers=config.NUM_WORKERS
    standardize_range=config.STANDARDIZE_RANGE
    downsample_size=config.DOWNSAMPLE_SIZE    
    type_save = config.TYPE_SAVE
    versions = config.TESTING_VERSION


    data_path = Path(data_root)
    images_path = str(data_path / "image") + "/"
    track_path = str(data_path / "track") + "/"
    metadata_path = str(data_path / "metadata.json")

    def image_filter(image):
        return (
            (image.grade() < 7)
            and (image.year() != 2023)
            and (100.0 <= image.long() <= 180.0)
        )  # and (image.mask_1_percent() <  self.corruption_ceiling_pct))

    def transform_func(image_ray):
        image_ray = np.clip(
            image_ray,standardize_range[0],standardize_range[1]
        )
        image_ray = (image_ray - standardize_range[0]) / (
            standardize_range[1] - standardize_range[0]
        )
        if downsample_size != (512, 512):
            image_ray = torch.Tensor(image_ray)
            image_ray = torch.reshape(
                image_ray, [1, 1, image_ray.size()[0], image_ray.size()[1]]
            )
            image_ray = nn.functional.interpolate(
                image_ray,
                size=downsample_size,
                mode="bilinear",
                align_corners=False,
            )
            image_ray = torch.reshape(
                image_ray, [image_ray.size()[2], image_ray.size()[3]]
            )
            image_ray = image_ray.numpy()
        return image_ray

    dataset = DigitalTyphoonDataset(
                str(images_path),
                str(track_path),
                str(metadata_path),
                "pressure",
                load_data_into_memory='all_data',
                filter_func=image_filter,
                transform_func=transform_func,
                spectrum="Infrared",
                verbose=False,
            )


    _,test_old = loading.load(0,dataset,batch_size,num_workers,type_save)
    _,test_recent = loading.load(1,dataset,batch_size,num_workers,type_save)
    _,test_now = loading.load(2,dataset,batch_size,num_workers,type_save)
    
    # Test
    
    trainer_old = pl.Trainer(
        logger=logger_old,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICE,
        max_epochs=config.MAX_EPOCHS,
        default_root_dir=config.LOG_DIR,
    )
    
    trainer_recent = pl.Trainer(
        logger=logger_recent,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICE,
        max_epochs=config.MAX_EPOCHS,
        default_root_dir=config.LOG_DIR,
    )

    trainer_now = pl.Trainer(
        logger=logger_now,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICE,
        max_epochs=config.MAX_EPOCHS,
        default_root_dir=config.LOG_DIR,
    )
    
    version_dir_old = 'tb_logs/resnet_train_old'
    version_dir_recent = 'tb_logs/resnet_train_recent'    
    version_dir_now = 'tb_logs/resnet_train_now'

    if type_save == 'same_size':
        version_dir_old += '_same'
        version_dir_recent += '_same'
        version_dir_now += '_same'
        
    
    
    
    with open("log.txt","a+") as file :
        file.write("\n------------------------------------------------------------ \n")
    for i in versions:
        
        with open("log.txt","a+") as file :
            file.write(f"\nVersion : {i} \n")
        version_path = f'/version_{i}/checkpoints/'
        _,_,filename_old = next(os.walk(version_dir_old + version_path))
        _,_,filename_recent = next(os.walk(version_dir_recent + version_path))
        _,_,filename_now = next(os.walk(version_dir_now+ version_path))
        model_old = LightningResnetReg.load_from_checkpoint(version_dir_old + version_path + filename_old[0])
        model_recent = LightningResnetReg.load_from_checkpoint(version_dir_recent + version_path + filename_recent[0])
        model_now = LightningResnetReg.load_from_checkpoint(version_dir_now + version_path + filename_now[0])
        
        print("Testing <2005")
        with open("log.txt","a+") as file : 
            file.write("Testing <2005 \n")  
        print("         on <2005 : ")
        trainer_old.test(model_old, test_old)
        print("         on >2005 : ")
        trainer_old.test(model_old, test_recent)
        print("         on >2015 : ")
        trainer_old.test(model_old, test_now)
        
        print("Testing >2005")
        with open("log.txt","a+") as file :
            file.write("Testing >2005\n")
        print("         on <2005 : ")
        trainer_recent.test(model_recent, test_old)    
        print("         on >2005 : ")
        trainer_recent.test(model_recent, test_recent)
        print("         on >2015 : ")
        trainer_recent.test(model_recent, test_now)
        
        print("Testing >2015")
        with open("log.txt","a+") as file : 
            file.write("Testing >2015\n")
        print("         on <2005 : ")
        trainer_now.test(model_now, test_old)
        print("         on >2005 : ")
        trainer_now.test(model_now, test_recent)
        print("         on >2015 : ")
        trainer_now.test(model_now, test_now)
        print(f"Run {i} done")
    

if __name__ == "__main__":
    main()
