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
import torch.nn.functional as Function
import random
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import itertools
import os

from model import EfficientNet_b0_baseline, EfficientNet_b0_Pretrained, EfficientNet_b0
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ['Surprise','Fear','Disgust','Happiness','Sadness','Anger','Neutral']

#https://github.com/javaidnabi31/Multi-class-with-imbalanced-dataset-classification/blob/master/20-news-group-classification.ipynb
def plot_confusion_matrix(args, cm, l_classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, 
                          recall = 0,
                          name = " "):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()
    fig.set_size_inches(14, 12, forward=True)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=recall)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(l_classes))
    plt.xticks(tick_marks, l_classes, rotation=90)
    plt.yticks(tick_marks, l_classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig('runs/' + args.model_name + '/' + args.model_name + name +  '-confusion.png')

def get_predictions(args, model, iterator, device):

    model.eval()
    model.to(device)
    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for train_batch_id, batch in enumerate(iterator):
            train_input = batch['image']
            train_label = batch['label']
            train_input = train_input.to(device)
            train_label = train_label.to(device)

            #y_pred, _ = model(x)
            y_pred = model(train_input)
            y_prob = Function.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)

            images.append(train_input.cpu())
            labels.append(train_label.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

    return images, labels, probs

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    image_min = img.min()
    image_max = img.max()
    img.clamp_(min=image_min, max=image_max)
    img.add_(-image_min).div_(image_max - image_min + 1e-5)

    #img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
        print("ONE CHANNEL!")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [Function.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


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

    model.to(device)
    model.eval()
    with torch.no_grad():
        for train_batch_id, batch in enumerate(val_dataloader):
            train_input = batch['image']
            train_label = batch['label']
            train_input = train_input.to(device)
            train_label = train_label.to(device)
            y_pred = model(train_input)
            loss = criterion(y_pred, train_label)
            acc = calculate_accuracy(y_pred, train_label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(val_dataloader), epoch_acc / len(val_dataloader)
    #return epoch_loss / 3., epoch_acc / 3.


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
    model.to(device)
    model.train()

    for batch_idx, batch in enumerate(train_dataloader):
        #model.model.eval()
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
        if batch_idx % args.log_interval == args.log_interval-1:
            val_loss, val_acc = evaluate(args, model, val_dataloader, criterion, device)
            model.train()
            if val_loss < best_valid_loss:
                print(f'Validation Loss Decreased({best_valid_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
                best_valid_loss = val_loss
                # Saving State Dict
                torch.save(model.state_dict(), 'runs/' + args.model_name + '/' + args.model_name + '.pth')

            args.train_loss.append(loss.item())
            args.val_loss.append(val_loss)
            args.val_acc.append(val_acc)
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\t'
                  'Train Loss: {:.2f} Train Acc: {:.2f}%  Validation Loss: {:.2f}  Validation Accuracy: {:.2f}%'.format(
                epoch, batch_idx * len(train_inputs), len(train_dataloader.dataset),
                       100. * batch_idx / len(train_dataloader), epoch_loss/args.log_interval, epoch_acc/args.log_interval, val_loss, val_acc))
            writer.add_scalar('training loss',
                            epoch_loss/args.log_interval,
                              (epoch-1) * len(train_dataloader) + batch_idx)
            writer.add_scalar('val accuracy',
                            val_acc,
                              (epoch-1) * len(train_dataloader) + batch_idx)
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(model, train_inputs, train_label),
                              global_step= (epoch - 1) * len(train_dataloader) + batch_idx)

            epoch_loss = 0.0
            epoch_acc = 0.0


    # return epoch_loss / len(train_dataloader), epoch_acc / len(train_dataloader)
    # return epoch_loss / 3., epoch_acc / 3.
    return best_valid_loss
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')


def initialize_parameters(m):
    #if isinstance(m, nn.Conv2d):
    #    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
    #    nn.init.constant_(m.bias.data, 0)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #nn.init.constant_(m.bias.data, 0)
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item["label"]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val["label"]]                                  
    return weight

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
    parser.add_argument('--continue-train', type=str,  default='NONE',
                        help='saves the current model')
    parser.add_argument('--transfer-train', type=str,  default='NONE',
                        help='saves the current model')
    parser.add_argument('--examine', default=False, action='store_true')

    args = parser.parse_args()
    # set seed

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
        T.RandomRotation(3),
        #T.RandomHorizontalFlip(0.5),
        # SquarePad(),
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    test_data_transform = T.Compose([
        T.ToPILImage("RGB"),
        # SquarePad(),
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_set = ImageDataset(train_labels, train_imgs_dir, transform=training_data_transform)
    val_set = ImageDataset(val_labels, val_imgs_dir, transform=test_data_transform)
    test_set = ImageDataset(test_labels, test_imgs_dir, transform=test_data_transform)


    print("trainset: ",len(train_set))
    print("val: ",len(val_set))
    print("testset: ",len(test_set))

    classes = ['Surprise','Fear','Disgust','Happiness','Sadness','Anger','Neutral']
    #weights = make_weights_for_balanced_classes(train_set, len(classes))                                                                
    #weights = torch.DoubleTensor(weights)                                       
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                    
    #train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle = True,                              
    #                                                         sampler = sampler, num_workers=args.workers, pin_memory=True)     

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True) #, sampler = sampler)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    # Load CIFAR10 dataset
    if(args.examine == True):
        model_old = EfficientNet_b0()
        model = EfficientNet_b0_Pretrained(model_old)
        model.load_state_dict(torch.load('runs/' + args.model_name + '/' + args.model_name+ '.pth'))
        #model = EfficientNet_b0_baseline()
        #model.load_state_dict(torch.load('runs/' + args.model_name + '/' + args.model_name+ '.pth'))
        model.to(device)

        images, labels, probs = get_predictions(args, model, test_dataloader, device)
        pred_labels = torch.argmax(probs, 1)
        cm = confusion_matrix(labels, pred_labels)
        #plot_confusion_matrix(args, labels, pred_labels)
        plot_confusion_matrix(args, cm, l_classes=np.asarray(classes), normalize=True,
                      title='Normalized recall confusion matrix', recall = 1, name = "-recall-")
        plot_confusion_matrix(args, cm, l_classes=np.asarray(classes), normalize=True,
                      title='Normalized precision confusion matrix', recall = 0, name = "-precision-")
        plot_confusion_matrix(args, cm, l_classes=np.asarray(classes), normalize=False,
                      title='Normalized confusion matrix', recall = 0, name = "-unnormalized-")


        print("done!")
    else:

        writer = SummaryWriter('runs/' + args.model_name)
        if(args.transfer_train != "NONE"):
            model_old = EfficientNet_b0()
            model_old.load_state_dict(torch.load('runs_pretrained/' + args.transfer_train + '/' + args.transfer_train + '-pretrained.pth')) #, strict=False)

            model = EfficientNet_b0_Pretrained(model_old)

        elif(args.continue_train == "NONE"):
            model = EfficientNet_b0_baseline()
            print("Train baseline!")
            #model.apply(initialize_parameters)
        else:

            model = EfficientNet_b0_baseline()
            model.load_state_dict(torch.load('runs/' + args.continue_train + '/' + args.continue_train + '.pth'))
            print("CONTINUE TRAIN MODE----")

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
        # opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Record loss and accuracy history
        args.train_loss = []
        args.val_loss = []
        args.val_acc = []

        # Train the model
        best_valid_loss = float('inf')

        for epoch in range(1, args.epochs + 1):
            start_time = time.monotonic()
            best_valid_loss = train(args, epoch, model, train_dataloader, val_dataloader, optimizer, criterion, device,
                                    writer, best_valid_loss)
            end_time = time.monotonic()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch :02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

        # Evaluate on test set
        writer.flush()

        #test time
        model = EfficientNet_b0_baseline()
        model.load_state_dict(torch.load('runs/' + args.model_name + '/' + args.model_name+ '.pth'))
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        loss, acc = evaluate(args, model, test_dataloader, criterion, device)
        print("TEST RESULTS: ", loss, acc)

if __name__ == '__main__':
    main()
