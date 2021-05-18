import os

import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_image
import json
from PIL import Image

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

        if(label_A == 2):
            label_A = 0
        elif(label_A == 0):
            label_A = 1

        sample = {"image_1": image_1, "image_2": image_2, "image_3": image_3, "label_A": label_A, "label_B": torch.FloatTensor(label_B)}

        return sample
