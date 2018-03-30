import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
from mydataset import OmniglotTrain
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import Siamese
import time


cuda = torch.cuda.is_available()


data_transforms = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ToTensor()
])


train_path = 'background'
train_dataset = dset.ImageFolder(root=train_path)

dataSet = OmniglotTrain(train_dataset, transform=data_transforms)

dataLoader = DataLoader(dataSet, batch_size=128,\
                        shuffle=True, num_workers=16)


def show_data_batch(sample_batched):
    image_batch = sample_batched[0][0]
    grid = torchvision.utils.make_grid(sample_batched)
    plt.figure()
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')
    plt.axis('off')
    plt.show()


#  def loss_fn(label, output):
    #  return -torch.mean(label * torch.log(output) + (1.0-label) * torch.log(1.0-output))
loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)


epochs = 200
learning_rate = 0.00006
momentum = 0.99


net = Siamese()

train_loss = []
net.train()
optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate )
optimizer.zero_grad()
if cuda:
    net.cuda()


for epoch in range(epochs):
    epoch_loss = 0
    for batch_id, (img1, img2, label) in enumerate(dataLoader):
        batch_start = time.time()
        if cuda:
            img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
        else:
            img1, img2, label = Variable(img1), Variable(img2), Variable(label)
        output = net.forward(img1, img2)
        loss = loss_fn(output, label)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        print('[%d/%d/%d]\tloss:\t%.5f\tTook\t%.2f s'%(batch_id, epoch+1, epochs, loss.data[0], time.time() - batch_start))
    train_loss.append(epoch_loss/(1.0+batch_id))
    torch.save(net.state_dict(), './model/model-epoch-%d.pth'%(epoch+1,))
    print("*"*30)
    #  learning_rate = learning_rate * 0.95

with open('train_loss', 'wb') as f:
    pickle.dump(train_loss, f)
