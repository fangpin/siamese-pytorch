import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image


class OmniglotTrain(Dataset):

    def __init__(self, dataset, transform=None):
        super(OmniglotTrain, self).__init__()
        np.random.seed(0)
        self.dataset = dataset
        self.transform = transform
        self.img1 = None

    def __len__(self):
        return  21000000

    def __getitem__(self, index):
        image1 = random.choice(self.dataset.imgs)
        # get image from same class
        label = None
        if index % 2 == 1:
            label = 1.0
            while True:
                image2 = random.choice(self.dataset.imgs)
                if image1[1] == image2[1]:
                    break
        # get image from different class
        else:
            label = 0.0
            while True:
                image2 = random.choice(self.dataset.imgs)
                if image1[1] != image2[1]:
                    break
        image1 = Image.open(image1[0])
        image2 = Image.open(image2[0])
        image1 = image1.convert('L')
        image2 = image2.convert('L')

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))



class OmniglotTest(Dataset):

    def __init__(self, dataset, transform=None, times=200, way=20):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.times = times
        self.way = way

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.img1 = random.choice(self.dataset.imgs)
            while True:
                img2 = random.choice(self.dataset.imgs)
                if self.img1[1] == img2[1]:
                    break
        # generate image pair from different class
        else:
            while True:
                img2 = random.choice(self.dataset.imgs)
                if self.img1[1] != img2[1]:
                    break

        img1 = Image.open(self.img1[0])
        img2 = Image.open(img2[0])
        img1 = img1.convert('L')
        img2 = img2.convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2


# test
if __name__=='__main__':
    omniglotTrain = OmniglotTrain('./images_background', 30000*8)
    print(omniglotTrain)
