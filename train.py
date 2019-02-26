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
import gflags
import sys
from collections import deque
import os


if __name__ == '__main__':

    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda", True, "use cuda")
    gflags.DEFINE_string("train_path", "/home/data/pin/data/omniglot/images_background", "training folder")
    gflags.DEFINE_string("test_path", "/home/data/pin/data/omniglot/images_evaluation", 'path of testing folder')
    gflags.DEFINE_integer("way", 20, "how much way one-shot learning")
    gflags.DEFINE_string("times", 400, "number of samples to test accuracy")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 128, "number of batch size")
    gflags.DEFINE_float("lr", 0.00006, "learning rate")
    gflags.DEFINE_integer("show_every", 10, "show result after each show_every iter.")
    gflags.DEFINE_integer("save_every", 100, "save model after each save_every iter.")
    gflags.DEFINE_integer("test_every", 100, "test model after each test_every iter.")
    gflags.DEFINE_integer("max_iter", 50000, "number of iterations before stopping")
    gflags.DEFINE_string("model_path", "/home/data/pin/model/siamese", "path to store model")
    gflags.DEFINE_string("gpu_ids", "0,1,2,3", "gpu ids used to train")

    Flags(sys.argv)

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])


    # train_dataset = dset.ImageFolder(root=Flags.train_path)
    # test_dataset = dset.ImageFolder(root=Flags.test_path)


    os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids
    print("use gpu:", Flags.gpu_ids, "to train.")

    trainSet = OmniglotTrain(Flags.train_path, transform=data_transforms)
    testSet = OmniglotTest(Flags.test_path, transform=transforms.ToTensor(), times = Flags.times, way = Flags.way)
    testLoader = DataLoader(testSet, batch_size=Flags.way, shuffle=False, num_workers=Flags.workers)

    trainLoader = DataLoader(trainSet, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)

    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
    net = Siamese()

    # multi gpu
    if len(Flags.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)

    if Flags.cuda:
        net.cuda()

    net.train()

    optimizer = torch.optim.Adam(net.parameters(),lr = Flags.lr )
    optimizer.zero_grad()

    train_loss = []
    loss_val = 0
    time_start = time.time()
    queue = deque(maxlen=20)

    for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
        if batch_id > Flags.max_iter:
            break
        if Flags.cuda:
            img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
        else:
            img1, img2, label = Variable(img1), Variable(img2), Variable(label)
        optimizer.zero_grad()
        output = net.forward(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()
        if batch_id % Flags.show_every == 0 :
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s'%(batch_id, loss_val/Flags.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        if batch_id % Flags.save_every == 0:
            torch.save(net.state_dict(), Flags.model_path + '/model-inter-' + str(batch_id+1) + ".pt")
        if batch_id % Flags.test_every == 0:
            right, error = 0, 0
            for _, (test1, test2) in enumerate(testLoader, 1):
                if Flags.cuda:
                    test1, test2 = test1.cuda(), test2.cuda()
                test1, test2 = Variable(test1), Variable(test2)
                output = net.forward(test1, test2).data.cpu().numpy()
                pred = np.argmax(output)
                if pred == 0:
                    right += 1
                else: error += 1
            print('*'*70)
            print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
            print('*'*70)
            queue.append(right*1.0/(right+error))
        train_loss.append(loss_val)
    #  learning_rate = learning_rate * 0.95

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    acc = 0.0
    for d in queue:
        acc += d
    print("#"*70)
    print("final accuracy: ", acc/20)
