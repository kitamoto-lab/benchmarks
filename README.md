# neurips2023-benchmarks

This repository contains the model and code used for benchmarking the Digital Typhoon Dataset as submitted to the NeuRIPS 2023 Dataset track. Contained are three folders: analysis, reanalysis, and forecasting, each one containing the code used for those respective benchmarks in the paper. 

Note: this repository originally lived in [this](https://github.com/jared-hwang/DigitalTyphoonModels) repository.


## Instructions to run

### Analysis Task

Two folder are available, one for the classification comparison with the results in the appendix and another one for the regression comparison.

To run the regression code :
```python
python3 train.py --model_name resnet18 --labels wind --size 224 --cropped True --device 0
```
all the parameters will 

To run the classification code (only available for wind label for now)
```python
python3 train.py --model_name vgg --size 224 --cropped True --device 0
```

### Forecasting

#### Docker
All of the below commands should be run in a Docker container built using the Dockerfile in the repo, with the data and repo being exposed as volumes in the container. 

To build:

```docker build  -t benchmarks_img .```

To run an interactive shell:

```docker run -it --shm-size=2G --gpus all -v /path/to/neurips2023-benchmarks:/neurips2023-benchmarks -v /path/to/datasets/:/data benchmarks_img```

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
python3 evaluate_foreacsting_pipeline.py
```

4. When the execution ends, it will print out statistics for absolute difference, percentage difference, and RMSE. Plots showing the expected and forecasted pressure will be saved into the directory ```Pipeline_logs/forecast_plots/```.