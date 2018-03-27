import torch
import torchvision
from torchvision import transforms
from mydataset import OmniglotTrain, ToTensor, RandomRotate
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

data_transforms = transforms.Compose([
    RandomRotate(-15, 15),
    ToTensor()
])


dataSet = OmniglotTrain('images_background', 270000,\
                        transform=data_transforms)

dataLoader = DataLoader(dataSet, batch_size=128,\
                        shuffle=True, num_workers=4)


def show_data_batch(sample_batched):
    image_batch = sample_batched[0][0]
    grid = torchvision.utils.make_grid(sample_batched)
    plt.figure()
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')
    plt.axis('off')
    plt.show()
