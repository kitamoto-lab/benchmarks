import config
import torch
from torch import nn
from pathlib import Path
import numpy as np
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
import random
import os

dataroot = config.DATA_DIR
batch_size=config.BATCH_SIZE
num_workers=config.NUM_WORKERS
split_by=config.SPLIT_BY
load_data=config.LOAD_DATA
dataset_split=config.DATASET_SPLIT
standardize_range=config.STANDARDIZE_RANGE
downsample_size=config.DOWNSAMPLE_SIZE
type_save=config.TYPE_SAVE

data_path = Path(dataroot)
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


years = dataset.get_years()
old=[]
recent=[]
now=[]

#splitting years in 3 buckets
for i in years :
    if i < 2005 :
        old.append(i)
    else :
        if i < 2015:
            recent.append(i)
        else :
            now.append(i)
            

old_data=[]
recent_data=[]
now_data=[]

#getting the ids from years
for year in old :
    old_data.extend(dataset.get_seq_ids_from_year(year))     
      
for year in recent :
    recent_data.extend(dataset.get_seq_ids_from_year(year))       
    
for year in now :
    now_data.extend(dataset.get_seq_ids_from_year(year))       

old_train , old_val = [],[]
recent_train , recent_val = [],[]
now_train , now_val = [],[]

#shuffling and splitting 80/20
random.shuffle(old_data)
random.shuffle(now_data)
random.shuffle(recent_data)

l=len(old_data)
for i in range(l):
    if i<l*0.8:
        old_train.append(old_data[i])
    else:
        old_val.append(old_data[i])
        
l=len(recent_data)
for i in range(l):
    if i<l*0.8:
        recent_train.append(recent_data[i])
    else:
        recent_val.append(recent_data[i])
          
l=len(now_data)
for i in range(l):
    if i<l*0.8:
        now_train.append(now_data[i])
    else:
        now_val.append(now_data[i])



#writting in file depending on which format
if(type_save=="standard"):
    if not(os.path.exists('./save')): os.mkdir('./save')
    with open('save/old_train.txt','w+') as file:
        for id in old_train:
            file.write(id+"\n")

    with open('save/old_val.txt','w+') as file:
        for id in old_val :
            file.write(id+"\n")

    with open('save/recent_train.txt','w+') as file:
        for id in recent_train:
            file.write(id+"\n")

    with open('save/recent_val.txt','w+') as file:
        for id in recent_val:
            file.write(id+"\n")

    with open('save/now_train.txt','w+') as file:
        for id in now_train:
            file.write(id+"\n")

    with open('save/now_val.txt','w+') as file:
        for id in now_val:
            file.write(id+"\n")

if(type_save=="same_size"):
    if not(os.path.exists('./save_same')): os.mkdir('./save_same')
    with(
        open('save_same/old_train.txt','w+') as train1,
        open('save_same/old_val.txt','w+') as test1,
        open('save_same/recent_train.txt','w+') as train2,
        open('save_same/recent_val.txt','w+') as test2,
        open('save_same/now_train.txt','w+') as train3,
        open('save_same/now_val.txt','w+') as test3,
    ):
        for i in range(min(len(old_train),len(recent_train),len(now_train))):
            train1.write(old_train[i]+'\n')
            train2.write(recent_train[i]+'\n')
            train3.write(now_train[i]+'\n')
        for i in range(min(len(old_val),len(recent_val),len(now_val))):            
            test1.write(old_val[i]+'\n')
            test2.write(recent_val[i]+'\n')
            test3.write(now_val[i]+'\n')

print("Saving Done !")        

