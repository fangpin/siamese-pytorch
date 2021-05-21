import torch
import torch.nn
import torch.nn.functional as F

"""
class ContrastiveLoss(torch.nn.Module):


    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x, y):
        x0 = x[:,0].reshape(-1, 1)
        x1 = x[:,1].reshape(-1, 1)
        self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

"""
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

def triplet_loss(y_pred):
    ref = y_pred[0::3, :]
    pos = y_pred[1::3, :]
    neg = y_pred[2::3, :]
    L12 = (ref - pos).pow(2).sum(1)
    L13 = (ref - neg).pow(2).sum(1)
    L23 = (pos - neg).pow(2).sum(1)
    correct = (L12 < L13) * (L12 < L23)

    alpha = 0.2
    d1 = F.relu((L12 - L13) + alpha)
    d2 = F.relu((L12 - L23) + alpha)
    d = torch.mean(d1 + d2)
    return d, torch.sum(correct)
