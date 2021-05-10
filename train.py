# Reference: https://github.com/pytorch/examples.
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataset_utils import ImageDataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import os

from model import Siamese
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def evaluate(args, model, val_dataloader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for train_batch_id, batch in enumerate(val_dataloader):
            print("hello", train_batch_id)
            train_input = batch['image']
            train_label = batch['label']
            train_input = train_input.to(device)
            train_label = train_label.to(device)
            y_pred = model(train_input)
            loss = criterion(y_pred, train_label)
            acc = calculate_accuracy(y_pred, train_label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            if train_batch_id >= 2:
                break
    # return epoch_loss / len(val_dataloader), epoch_acc / len(val_dataloader)
    return epoch_loss / 3., epoch_acc / 3.


def train(args, epoch, model, train_dataloader, val_dataloader, optimizer, criterion, device, writer, best_valid_loss):
    """
    Train the model for one epoch.
    Arguments:
        args: training settings
        epoch: epoch index
        model: model in one of ('softmax', 'twolayernn','convnet')
        train_loader: training data loader
        val_loader: validation data loader
        test_loader: test data loader
        loss_func: loss function, which is cross entropy loss in this repository
        opt: optimizer
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch_idx, batch in enumerate(train_dataloader):
        print("hello", batch_idx)
        train_inputs = batch['image']
        train_label = batch['label']
        train_inputs = train_inputs.to(device)
        train_label = train_label.to(device)
        optimizer.zero_grad()
        y_pred = model(train_inputs)
        loss = criterion(y_pred, train_label)
        acc = calculate_accuracy(y_pred, train_label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        if batch_idx % args.log_interval == 0:
            val_loss, val_acc = evaluate(args, model, val_dataloader, criterion, device)
            if val_loss < best_valid_loss:
                print(f'Validation Loss Decreased({best_valid_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
                best_valid_loss = val_loss
                # Saving State Dict
                torch.save(model.state_dict(), args.model_name+ '.pth')

            args.train_loss.append(loss.item())
            args.val_loss.append(val_loss)
            args.val_acc.append(val_acc)
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\t'
                  'Train Loss: {:.2f}  Validation Loss: {:.2f}  Validation Accuracy: {:.2f}%'.format(
                epoch, batch_idx * len(train_inputs), len(train_dataloader.dataset),
                       100. * batch_idx / len(train_dataloader), loss.item(), val_loss, val_acc))
            writer.add_scalar('training loss',
                            epoch_loss/args.log_interval,
                              (epoch-1) * 3 + batch_idx)
            writer.add_scalar('val accuracy',
                            val_acc,
                              (epoch-1) * 3 + batch_idx)
            epoch_loss = 0.0

        if batch_idx >= 2:
            break
    # return epoch_loss / len(train_dataloader), epoch_acc / len(train_dataloader)
    # return epoch_loss / 3., epoch_acc / 3.

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')


def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='EMOTION CLASSIFICATION')
    parser.add_argument('--batch-size', type=int, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--dataset-dir', default='data',
                        help='directory that contains cifar-10-batches-py/ '
                             '(downloaded automatically if necessary)')
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--log-interval', type=int, default=75, metavar='N',
                        help='number of batches between logging train status')
    parser.add_argument('--lr', type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--model-name', type=str,  default='run-01',
                        help='saves the current model')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay hyperparameter')
    args = parser.parse_args()
    # set seed
    writer = SummaryWriter('runs/' + args.model_name)

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    train_imgs_dir = os.path.join(args.dataset_dir, "train")
    train_labels = pd.read_csv(os.path.join(args.dataset_dir, "label/train_label.csv"))

    val_imgs_dir = os.path.join(args.dataset_dir, "val")
    val_labels = pd.read_csv(os.path.join(args.dataset_dir, "label/val_label.csv"))

    test_imgs_dir = os.path.join(args.dataset_dir, "test")
    test_labels = pd.read_csv(os.path.join(args.dataset_dir, "label/test_label.csv"))

    training_data_transform = T.Compose([
        T.ToPILImage("RGB"),
        T.RandomRotation(5),
        T.RandomHorizontalFlip(0.5),
        # SquarePad(),
        T.Resize(128),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    test_data_transform = T.Compose([
        T.ToPILImage("RGB"),
        # SquarePad(),
        T.Resize(128),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_set = ImageDataset(train_labels, train_imgs_dir, transform=training_data_transform)
    val_set = ImageDataset(val_labels, val_imgs_dir, transform=test_data_transform)
    test_set = ImageDataset(test_labels, test_imgs_dir, transform=test_data_transform)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    # Load CIFAR10 dataset
    n_classes = 7

    model = Siamese()
    model.apply(initialize_parameters)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    criterion = criterion.to(device)
    model.train()
    optimizer.zero_grad()

    # Define optimizer
    #opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Record loss and accuracy history
    args.train_loss = []
    args.val_loss = []
    args.val_acc = []

    # Train the model
    best_valid_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        start_time = time.monotonic()

        best_valid_loss = train(args, epoch, model, train_dataloader, val_dataloader, optimizer, criterion, device, writer, best_valid_loss)
        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch :02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

    # Evaluate on test set
    writer.flush()

    model = Siamese()
    model.load_state_dict(torch.load(args.model_name+ '.pth'))
    loss, acc = evaluate(args, model, test_dataloader, criterion, device)
    print("TEST RESULTS: ", loss, acc)



if __name__ == '__main__':
    main()
