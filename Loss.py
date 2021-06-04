import torch.nn as nn
import torch
import numpy
import torch.nn.functional as F


class Multi_cross_entropy(nn.Module):
    def __init__(self):
        super(Multi_cross_entropy, self).__init__()

    def forward(self, pred, soft_targets):

        logsoftmax = nn.LogSoftmax()
        return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), 1))



def total_loss(y_pred_A, train_label_A, pred_B, soft_targets_B):
    task_A_loss = nn.CrossEntropyLoss(y_pred_A, train_label_A)
    task_B_loss = cross_entropy(pred_B, soft_targets_B)

    return task_A_loss + task_B_loss

def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), 1))

class triplet_loss(nn.Module):
    def __init__(self):
        super(triplet_loss, self).__init__()

    def forward(self, out1, out2, out3):
        ref = out1
        pos = out2
        neg = out3
        L12 = (ref - pos).pow(2).sum(1)
        L13 = (ref - neg).pow(2).sum(1)
        L23 = (pos - neg).pow(2).sum(1)
        correct = (L12 < L13) * (L12 < L23)
        #print(torch.sum(correct)/out1.size(0))

        alpha = 0.2
        d1 = F.relu((L12 - L13) + alpha)
        d2 = F.relu((L12 - L23) + alpha)
        d = torch.mean(d1 + d2)
        return d, 100*torch.sum(correct)/out1.size(0)
    

