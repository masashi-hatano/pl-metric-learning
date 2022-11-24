<div align="center">

# Pytorch-Lightning Metric Learning

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
  
  In this repository, we implemented metric learning for cifar-10 classification by using hydra+pytorch-lightning.
  
</div>

<p align="center">
  <img src="https://user-images.githubusercontent.com/71377772/182391423-a6bd7d65-4a6b-4584-96bb-e87b8df5c713.gif" width=75% height=75% />
</p>

## cifar-10 classification with ResNet+ArcFace

### configs
You can manage hyparams for deep learning thanks to hydra!  
You need to change "data_dir", which is a directory for downloaded cifar-10 dataset, if this directory contains nothing, dataset will automatically be downloaded there. 

### data_module
Here, you can change dataset and data loader behaviour.  
Normally, you don't need to change anything here.

### model
lit_MetricTrainer is the most important part of the implementation as it defines each step of train,val, and test.

## How to run the code
First, you need to create the virtural environment.
```
python -m venv pl-metric
```
And then, activate the virtural environment.  
(for Windows)
```
./pl-metric/Scripts/activate
```
(for Mac or Linux)
```
source pl-metric/bin/activate
```

Second, you need to install libraries to execute the code.
```
pip -r install requirements.txt
```

Finally, you are ready to run the code!  
```
python lit_main.py
```
