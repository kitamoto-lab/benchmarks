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

