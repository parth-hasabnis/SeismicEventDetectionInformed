# Urban Seismic Event Detection

This file will serve as a tutorial to train and test AI models to perform Urban Seismic Event Detection 

## Requirements
The following python packages are necessary to create a dataset and train a model (Versions included)
```
python=3.9
numpy=1.23.5
librosa=0.10.0
obspy=1.4.0
matplotlib=3.7.1
scipy=1.11.3
scikitlearn=1.3.1
torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pytorch version=1.31.1
```
The code has been tested on the following versions as well
```
python=3.9
numpy=1.24.4
librosa=0.10.1
obspy=1.4.0
matplotlib=3.9.0
scipy=1.11.3
scikitlearn=1.3.1
torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pytorch version=2.1.0
```
## Downloading the Dataset

The dataset can be found at: [https://zenodo.org/doi/10.5281/zenodo.10724592](https://zenodo.org/doi/10.5281/zenodo.10724592). 
Contains Datasets for training and testing models for Urban Seismic Event Detection (USED).

1. Strong Dataset: Contains Synthetic Data to be used for supervised learning
2. Unlabel Dataset: Contains unlabeled data to be used for semi-supervised (or unsupervised) learning
3. Test Synth: Synthetic Dataset to evaluate models
4. Test Real: Small Real Dataset to evaluate model

## Training and Evaluation

Run the following command to train the model:
```
python train.py -f [training-arguments]
```
The command line argument is a JSON file with all training argument values. The default training arguments are described in "default_arguments.json". Eg:
```
python train.py -f default_arguments.json
```
To evaluate the performance of the model on labelled datasets, run the following command:
```
python evaluate.py -f [eval-arguments] -r [bool] -m [bool] -p [bool]
```
It takes the following arguments:
--file     -f    Testing arguments. Eg: default_eval_arguments.json
--roc      -roc  Generate Receiver Operating Characteristic. default: True
--metrics  -m    Generate evaluation metrics. default: True
--plot     -p    Save spectrograms of input data overlayed with labels and predictions. default: False

Eg: ``` python evaluate.py -f default_eval_arguments.json -p ```

To test the model on long-term continuous data, run the following command:
```
python test.py -f [test-arguments] -p [bool]
```
It takes the following arguments:
--file     -f    Testing arguments. Eg: default_eval_arguments.json
--plot     -p    Save spectrograms of input data overlayed with labels and predictions. default: False

Eg: ``` python test.py -f default_test_arguments.json -p ```
