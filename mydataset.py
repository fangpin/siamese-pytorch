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
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset.imgs)

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


class OmniglotTest(object):

    def __init__(self, data_path, way=20):
        super(OmniglotTest, self).__init__()
        self.way = way
        self.samples = []
        tmp = []
        path = [os.path.join(data_path, p) for p in os.listdir(data_path)]
        alphabetas = npc(path, 10)
        for time in range(2):
            for i, alpha in enumerate(alphabetas):
                cnt = 0
                for j, char in enumerate([os.path.join(alpha, p) for p in os.listdir(alpha)]):
                   cnt += 1
                   # print(os.path.join(char, p), 2)
                   tmp.append(npc([os.path.join(char, p) for p in os.listdir(char)], 2))
                   if cnt == self.way:
                       break

        for i in range(20):
            for j in range(self.way):
                for k in range(self.way):
                    idx1, idx2 = i * self.way + j, i * self.way + k
                    label = 1.0 if idx1 == idx2 else 0.0
                    self.samples.append(((tmp[idx1][0], tmp[idx2][1]), label))



# test
if __name__=='__main__':
    # omniglotTrain = OmniglotTrain('./images_background', 30000*8)
    # print(omniglotTrain)
    test = OmniglotTest('./images_evaluation', 20)
