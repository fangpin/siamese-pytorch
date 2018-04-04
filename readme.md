# Siamese Networks for One-Shot Learning

A reimplementation of the [original paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) by pytorch.
train and test on the dataset Omniglot

## requirment
- pytorch
- torchvision
- python3.5+

## run step
- download dataset
- preprocess dataset to make dataLoader easier by
``` shell
python3 make_dataset.py
```
- train and test by
```shell
python3 train.py
```

## experiment result
loss value sampled after each 200 batches
![img](https://github.com/fangpin/siamese-network/blob/master/loss.png)
My final precision is 89.5% a little smaller than the result of the papers(92%)

The little result difference might caused by some difference between my implmentation and the paper's. I list these differences as following:

- learning rate
instead of using SGD with momentum I just use adam simply.
- paraments initialization
instead of using individual initialization method, learning rate and regularization rate in different layer I simply use the 
default setting of pytroch and keep them same.
