# benchmarks

This repository contains the model and code used for benchmarks of the Digital Typhoon Dataset. Contained are three folders: analysis, reanalysis, and forecasting, each one containing the code used for those respective tasks. 

Note: this repository originally lived in [this](https://github.com/jared-hwang/DigitalTyphoonModels) repository.


## Instructions to run

#### Docker
All of the below commands should be run in a Docker container built using the Dockerfile in the repo, with the data and repo being exposed as volumes in the container. 

To build:

```docker build  -t benchmarks_img .```

To run an interactive shell:

```docker run -it --shm-size=2G --gpus all -v /path/to/neurips2023-benchmarks:/neurips2023-benchmarks -v /path/to/datasets/:/data benchmarks_img```


### Analysis Task

In the analysis part, two folder are available, one for the classification comparison with the results in the appendix and another one for the regression comparison in the benchmark section (section 5.1).

First, the path to the dataset must be correctly initialized in the config.py file in analysis/regression and analysis/classification, the variable DATA_DIR must lead to the dataset with the folders images/, metadata/ and the file metadata.json.

To run the code for regression or classification :
```bash
python3 train.py --model_name resnet18 --size 224 --cropped True --device 0
```

Replace the arguments with your desired values:

- `--device`: The index of the GPU device to use (0 by default) (e.g. number between 0 and the number of devices in your GPU, run ```nvidia-smi``` for more infos).
- `--size`: Input image size (e.g., 512).
- `--cropped`: Whether to use cropped images (True or False).

The train.py files are the entry points of the code. And then it basically follows the PytorchLightning workflow :
https://lightning.ai/docs/pytorch/stable/starter/introduction.html

#### Regression
More specifically to run the regression code :
```bash
cd analysis/regression
python3 train.py --model_name resnet18 --labels wind --size 224 --cropped True --device 0
```
In this case more arguments can be changed :
- `--model_name`: The architecture of the model (e.g., "resnet18", "resnet50").
- `--label`: The target label for regression (e.g., "wind", "pressure").

By changing the arguments you can obtain the results of the tables 2 and 3 in the paper.

A pipeline is also provided which trains all the benchmarks in a row:
```bash
cd analysis/regression
python3 pipeline.py --device 0
```

#### Classification
To run the classification code :
```bash
cd analysis/classification
python3 train.py --model_name vgg --size 224 --cropped True --device 0
```
In this case, the model_name possibilities are different :
- `--model_name`: The architecture of the model (e.g., "resnet18", "vgg", "vit").

Same as above, a pipeline in the classification folder launch all the training of the classification benchmarks in a row:
```bash
cd analysis/classification
python3 pipeline.py --device 0
```

#### Results visualisation
The results are saved in a /analysis/[classification,regression]/results/

Folders with names depending of the arguments are created after a training and it creates different version after a few tries. It is managed by the pytorch_lightning.loggers TensorboardLogger : https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.TensorBoardLogger.html#tensorboardlogger

To visualize the results you can launch the command 
```bash
tensorboard --logdir /results/[training_name]/version_X/
```
And then open the local link given in a browser.


### Forecasting

Ensure that when running the following commands, the appropriate path to ```WP/``` is specified in the ```hyperparameters.py``` files in ```ConvLSTM/``` and ```ResNet/``` in the variable ```data_dir```.

To train and run the pipeline, two models must be trained: 

1. First, the convolutional LSTM.

2. Then, using the trained convLSTM, the ResNET.

#### ConvLSTM

Instructions to train the convLSTM are as follows:

1. Enter the directory ```forecasting```

2. Run 
```
python3 train_convLSTM.py
```

3. Logs and checkpoints will be saved to ```ConvLSTM_logs/lightning_logs/version_[i]```. This is where validation indices are also saved in ```validation_indices.txt```

#### Pressure Regression ResNet

Instructions to train the ResNet are as follows:

1. Enter the directory ```forecasting```

2. Two paths must be set in the ```train_resnet.py``` file. On line 32, the path to the convLSTM saved model must be specified. Similarly, on line 168, the path to the file specifying what indices are validation indices must be specified.

3. Run 
```
python3 train_resnet.py
```

4. Logs and checkpoints will be saved to ```ResNet_logs/lightning_logs/version_[i]```. 

#### Evaluation

Instructions to evaluate the pipeline and produce RMSE, difference statistics as well as plots are as follows:

1. Enter the directory ```forecasting```

2. Three paths must be set in ```evaluate_forecasting_pipeline.py```. On line 150, set the path to the dataset indices used as validation indices during the convLSTM training. on Line 153, set the path to the convLSTM model weights. On line 156, set the path to the ResNet model weights.

3. Run 
```
python3 evaluate_forecasting_pipeline.py
```

4. When the execution ends, it will print out statistics for absolute difference, percentage difference, and RMSE. Plots showing the expected and forecasted pressure will be saved into the directory ```Pipeline_logs/forecast_plots/```.

### Reanalysis Task
Every command should be run in the reanalysis folder. The path to this folder and to the data should be provided in the config.py file.

#### Create buckets
First, you have to split and save the dataset into 3 buckets according to the type of splitting refered in the config.py file ('standard' for standard splitting between before 2005 / between 2005 and 2015 / after 2015, 'same_size' for the same splitting but with a equal number of sequences per bucket).
```
python3 createdataset.py
```
This will create a folder (named 'save' or 'save_same') with 6 .txt file containing the id of the sequences used for training and testing in each bucket.

#### Train
You can now train for a number of runs (called version in the logs) and epochs specified in the config.py file.
```
python3 train_split.py
```
A tensorboard log while be created for each run with each bucket in the tb_logs.

#### Test
After specifing a list of versions in the config.py file, you'll be able to test the model.
```
python3 split_testing.py
```
The accuracy (RMSE in hPa) will be displayed on the terminal but also written in a log.txt file in the directory ```reanalysis```.
