# Siamese Networks for One-Shot Learning

A reimplementation of the [original paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) in pytorch with
training and testing on the [Omniglot dataset](https://github.com/brendenlake/omniglot).

## requirement
- pytorch
- torchvision
- python3.5+
- python-gflags

See requirements.txt 

## run step
- download dataset
```
git clone https://github.com/brendenlake/omniglot.git
cd omniglot/python
unzip images_evaluation.zip
unzip images_background.zip
cd ../..
# setup directory for saving models
mkdir models
```
- train and test by running
```shell
python3 train.py --train_path omniglot/python/images_background \
                 --test_path  omniglot/python/images_evaluation \
                 --gpu_ids 0 \
                 --model_path models
```

## experiment result
Loss value is sampled after every 200 batches
![img](https://github.com/fangpin/siamese-network/blob/master/loss.png)
My final precision is 89.5% a little smaller than the result of the paper (92%).

The small result difference might be caused by some difference between my implementation and the paper's. I list these differences as follows:

- learning rate

instead of using SGD with momentum I just use ADAM.

- parameters initialization and settings

Instead of using individual initialization methods, learning rates and regularization rates at different layers I simply use the default setting of pytorch and keep them same.
