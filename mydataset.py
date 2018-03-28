import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
from skimage import io
from skimage import transform as sk_transform
import numpy as np
from torchvision import transforms
import time
import random


class OmniglotTrain(Dataset):

    def __init__(self, data_path, size, transform=None):
        print('making dataset')
        start = time.time()
        super(OmniglotTrain, self).__init__()
        self.size = size
        self.transform = transform
        alphabetas = [os.path.join(data_path, p) for p in os.listdir(data_path)]
        characters = []
        self.samples = []
        for alpha in alphabetas:
            characters.extend([os.path.join(alpha, p) for p in os.listdir(alpha)])
        for i in range(size):
            # produce data pairs from same characters
            if i%2 == 1:
                char = npc(characters, 1)[0]
                pair = npc([os.path.join(char, p) \
                            for p in os.listdir(char)], 2)
                self.samples.append((pair, 1.0))
            # produce data pairs from different characters
            else:
                chars = npc(characters, 2)
                p0 = npc([os.path.join(chars[0], p)\
                            for p in os.listdir(chars[0])], 1)[0]
                p1 = npc([os.path.join(chars[1], p)\
                            for p in os.listdir(chars[1])], 1)[0]
                self.samples.append(((p0, p1), 0.0))
        end = time.time()
        print('finishing making dataset.\tTook:%.2f s'%(end-start,))
        print('*'*30)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image1 = io.imread(self.samples[index][0][0])
        image2 = io.imread(self.samples[index][0][1])
        image1 = image1[np.newaxis, :]
        image2 = image2[np.newaxis, :]
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return (image1, image2), torch.from_numpy(np.array(self.samples[index][1]))


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


class RandomRotate(object):

    def __init__(self, min, max):
        # random rotate at degree between [min, max]
        self.min = min
        self.max = max

    def __call__(self, img):
        img = sk_transform.rotate(img[0, :], random.randint(self.min, self.max), mode='edge')
        return img[np.newaxis, :]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


# test
if __name__=='__main__':
    # omniglotTrain = OmniglotTrain('./images_background', 30000*8)
    # print(omniglotTrain)
    test = OmniglotTest('./images_evaluation', 20)
