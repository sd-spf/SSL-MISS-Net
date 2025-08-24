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

Organize the folder as follows:

```
|-- dataset/
|   |-- train/
|   |   |-- class1
|   |   |   |-- 32
|   |   |   |-- 128
|   |   |-- class1
|   |   |   |-- 32
|   |   |   |-- 128
|   |   |-- ...
|   |-- test/
|   |   |-- image1.png
|   |   |-- image2.png
...
```

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
