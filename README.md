# SSL-MISS-Net
An Integrated MRI-Based Diagnostic Framework for Glioma with Incomplete Imaging Sequences and Imperfect Annotations
![Model](./fig1.png)

# Software Requirements

## Hardware requirements

The package development version is tested on Linux operating systems.

Linux: Ubuntu 16.04

window: window 10 

CUDA/cudnn:10.1

## Python Dependencies
```
> - Python
> - PyTorch-cuda
> - torchvision
> - opencv
> - numpy
> - json
> - os

### Prepare dataset

1. Prepare an Excel file containing patient names and the paths to complete imaging sequences as the data input for self-supervised learning (SSL) training.

2. Prepare another Excel file containing patient names, the paths to two imaging sequences, and the corresponding prediction labels as the data input for supervised prediction training. For missing imaging sequences, use None; for missing labels, use -1.

# Training and Evaluation example

> Training and evaluation are on a single GPU.

### PreTrain(Self-Supervised Learning train)

```
python pretrain.py
```
### Train

```
python train.py
```

### Evaluation
```
python test.py
```
