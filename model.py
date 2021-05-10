import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):

    #TODO: Add Batch Norm
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 94),  # 96x35x35
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride = 2),  # 96x17x17
            nn.Conv2d(96, 256, 7, stride = 1, padding= 3),
            nn.ReLU(),    # 256x17x17
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride = 2),   # 256x8x8
            nn.Conv2d(256, 384, 5, padding = 2, stride = 1), #384x8x8
            nn.ReLU(), # 128@18*18
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, 5, padding=2, stride=1), #256x8x8
            nn.ReLU(),  # 128@18*18
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride = 2), # 256x4x4
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 8))

        #self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        #self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

    def forward(self, x1):#, x2):
        out1 = self.forward_one(x1)
        #out2 = self.forward_one(x2)
        #dis = torch.abs(out1 - out2)
        #out = self.out(dis)
        #  return self.sigmoid(out)
        return out1

# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))
