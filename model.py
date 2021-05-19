import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):

    #TODO: Add Batch Norm
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 94),  # 96x35x35
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

            nn.Linear(4096, 7))

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

class Siamese(nn.Module):

    #TODO: Add Batch Norm
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 94),  # 96x35x35
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
            nn.ReLU(inplace=True))

        self.task_A = nn.Sequential(
            #nn.Linear(12288, 4096),
            #nn.BatchNorm1d(4096),
            #nn.Dropout(0.5),
            #nn.ReLU(inplace=True),

            #nn.Linear(4096, 4096),
            #nn.BatchNorm1d(4096),
            #nn.Dropout(0.5),
            #nn.ReLU(inplace=True),

            #nn.Linear(4096, 4096),
            #nn.BatchNorm1d(4096),
            #nn.Dropout(0.5),
            #nn.ReLU(inplace=True),

            nn.Linear(4096, 1)
        )
        self.task_A_concat = nn.Sequential(
            #nn.Linear(12288, 4096),
            #nn.BatchNorm1d(4096),
            #nn.Dropout(0.5),
            #nn.ReLU(inplace=True),

            #nn.Linear(4096, 4096),
            #nn.BatchNorm1d(4096),
            #nn.Dropout(0.5),
            #nn.ReLU(inplace=True),

            #nn.Linear(4096, 4096),
            #nn.BatchNorm1d(4096),
            #nn.Dropout(0.5),
            #nn.ReLU(inplace=True),

            nn.Linear(3, 2)
        )
    """
        self.task_B = nn.Sequential(
            nn.Linear(12288, 4096),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 3)
        )

    """

    def max_pooling_FC(self, x1, x2, x3):
        max_1 = torch.maximum(x1, x2)
        max = torch.maximum(max_1, x3)
        return max

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

    def forward_task_A(self, x1,x2,x3):
        #dis12 = torch.pow(x1-x2, 2)
        #dis23 = torch.pow(x2-x3, 2)
        #dis13 = torch.pow(x1-x3, 2)
        dis12 = torch.abs(x1-x2)
        dis23 = torch.abs(x2-x3)
        dis13 = torch.abs(x1-x3)
        #max12 = torch.maximum(x1, x2)
        #max23 = torch.maximum(x2, x3)
        #max13 = torch.maximum(x1, x3)
        #concat = torch.cat((torch.tensor(max12), torch.tensor(max23), torch.tensor(max13)), 1)
        x12 = self.task_A(dis12)
        x23 = self.task_A(dis23)
        x13 = self.task_A(dis13)
        #concat = torch.cat((torch.tensor(x12), torch.tensor(x23), torch.tensor(x13)), 1)

        #out = self.task_A_concat(concat)

        return x12,x23,x13

    def forward_task_B(self, x):
        x = self.task_B(x)
        return x


    def forward(self, x):#, x2):
        x1,x2,x3 = x

        #print(x1.size())
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        out3 = self.forward_one(x3)

        #concat = torch.cat((torch.tensor(out1), torch.tensor(out2), torch.tensor(out3)), 1)

        x12,x23,x13 = self.forward_task_A(out1, out2, out3)
        task_A_out = (x12,x23,x13)
        #task_B_out = self.forward_task_B(concat)

        #return task_A_out, task_B_out
        return task_A_out

# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f'The model has {count_parameters(net):,} trainable parameters')
