import torch.nn as nn
import torch
import numpy


class Multi_cross_entropy(nn.Module):
    def __init__(self):
        super(Loss_A, self).__init__()

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

