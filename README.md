# Classification_with_Cleanlab
  

## Introduction
Classification_with_Cleanlab is an open source image classification toolbox using Cleanlab based on PyTorch.

The toolbox efficiently utilize Cleanlan to learn with noisy labels and finding label errors in datasets.

Thanks for the author's (@Curtis G. Northcutt) awesome work--Cleanlab!
For more details about Cleanlab, please see the original paper: 
[Confident Learning: Estimating Uncertainty in Dataset Labels](https://arxiv.org/abs/1911.00068) | [blog](https://l7.curtisnorthcutt.com/confident-learning)

## Major features

- High efficient cross validation and training
- Various backbones and pretrained models
- Large-scale training configs
- High efficiency and extensibility


## Requirements

This is my experiment eviroument
- python3.6
- pytorch1.6.0+cu101

## Dataset
Supported:
- [x] CIFAR10
- [x] cifar10
- [x] ImageNet

## Benchmark 

Supported backbones:
- [x] VGG
- [x] ResNet
- [x] ResNeXt
- [x] SE-ResNet
- [x] SE-ResNeXt
- [x] Densenet
- [x] RegNet
- [x] ShuffleNetV1
- [x] ShuffleNetV2
- [x] MobileNetV2
- [x] MobileNetV3
- [x] Efficientnet b0-b7



## Train

```bash
# use gpu to train vgg16
$ python Train_CIFAR.py -Backbone vgg16 -Datasets cifar10
```

## Test 
Test the model using eval.py


## Results
All models are trained in the same condition, and might not get the best result


More experiment results coming soon

## Contact

Eamil: wwj123@zju.edu.cn