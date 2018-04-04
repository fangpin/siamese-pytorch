import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
from mydataset import OmniglotTrain, OmniglotTest
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import Siamese
import time
import numpy as np


cuda = torch.cuda.is_available()


data_transforms = transforms.Compose([
    transforms.RandomAffine(15),
    transforms.ToTensor()
])


train_path = 'background'
test_path = 'evaluation'
train_dataset = dset.ImageFolder(root=train_path)
test_dataset = dset.ImageFolder(root=test_path)

way = 20
times = 400

dataSet = OmniglotTrain(train_dataset, transform=data_transforms)
testSet = OmniglotTest(test_dataset, transform=transforms.ToTensor(), times = times, way = way)
testLoader = DataLoader(testSet, batch_size=way, shuffle=False, num_workers=16)

dataLoader = DataLoader(dataSet, batch_size=128,\
                        shuffle=False, num_workers=16)


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


learning_rate = 0.00006


net = Siamese()

train_loss = []
net.train()
optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate )
optimizer.zero_grad()
if cuda:
    net.cuda()

show_every = 10
save_every = 100
test_every = 100
train_loss = []
loss_val = 0
max_iter = 90000

for batch_id, (img1, img2, label) in enumerate(dataLoader, 1):
    if batch_id > max_iter:
        break
    batch_start = time.time()
    if cuda:
        img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
    else:
        img1, img2, label = Variable(img1), Variable(img2), Variable(label)
    optimizer.zero_grad()
    output = net.forward(img1, img2)
    loss = loss_fn(output, label)
    loss_val += loss.data[0]
    loss.backward()
    optimizer.step()
    if batch_id % show_every == 0 :
        print('[%d]\tloss:\t%.5f\tTook\t%.2f s'%(batch_id, loss_val/show_every, (time.time() - batch_start)*show_every))
        loss_val = 0
    if batch_id % save_every == 0:
        torch.save(net.state_dict(), './model/model-batch-%d.pth'%(batch_id+1,))
    if batch_id % test_every == 0:
        right, error = 0, 0
        for _, (test1, test2) in enumerate(testLoader, 1):
            if cuda:
                test1, test2 = test1.cuda(), test2.cuda()
            test1, test2 = Variable(test1), Variable(test2)
            output = net.forward(test1, test2).data.cpu().numpy()
            pred = np.argmax(output)
            if pred == 0:
                right += 1
            else: error += 1
        print('*'*70)
        print('[%d]\tright:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
        print('*'*70)
    train_loss.append(loss_val)
#  learning_rate = learning_rate * 0.95

with open('train_loss', 'wb') as f:
    pickle.dump(train_loss, f)
