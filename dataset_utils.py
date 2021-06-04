import os

import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_image
import json
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, labels, img_dir, transform=None, target_transform=None):
        self.img_labels = labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        sample = {"image": torch.tensor(image, dtype=torch.float32), "label": label - 1}

        return sample

class PretrainImageDataset(Dataset):
    def __init__(self, labels, img_dir, transform=None, target_transform=None):
        self.img_labels = labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_1_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img_2_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        img_3_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 2])

        #image_1 = read_image(img_1_path)
        image_1 = Image.open(img_1_path)
        image_1 = image_1.convert("RGB")

        image_2 = Image.open(img_2_path)
        image_2 = image_2.convert("RGB")

        image_3 = Image.open(img_3_path)
        image_3 = image_3.convert("RGB")


        #image_2 = read_image(img_2_path)
        #image_3 = read_image(img_3_path)

        label_A = self.img_labels.iloc[idx, 3]
        label_B = self.img_labels.iloc[idx, 4]
        label_B = json.loads(label_B)

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image_3 = self.transform(image_3)

        #if self.target_transform:
        #    label = self.target_transform(label)
        """
        if(label_A == 2):
            label_A = 1.
        elif(label_A == 0):
            label_A = 0.
            
        


        sample = {"image_1": image_1, "image_2": image_2, "image_3": image_3, "label_A": torch.from_numpy(np.array([label_A],dtype=np.float32)), "label_B": torch.FloatTensor(label_B)}
        """
        images = [image_1, image_2, image_3]
        label_B = np.array(label_B)
        argm = np.argmax(label_B)
        img3 = images.pop(argm)

        img1 = images[0]
        img2 = images[1]


        sample = {"image_1": img1, "image_2": img2, "image_3": img3, "label_A": label_A, "label_B": torch.FloatTensor(label_B), "name1": self.img_labels.iloc[idx, 0],"name2": self.img_labels.iloc[idx, 1],"name3": self.img_labels.iloc[idx, 2] }

        return sample
