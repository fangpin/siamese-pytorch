import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import timm


class EfficientNet_b0(nn.Module):
    def __init__(self):
        super(EfficientNet_b0, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.classifier_layer = nn.Sequential(
            nn.Linear(1280 , 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 , 16),
            nn.ReLU()
        )
        #torch.nn.init.xavier_uniform_(self.classifier_layer[0].weight)
        #torch.nn.init.xavier_uniform_(self.fc[0].weight)

    def forward(self, x):

        out1 = self.model.extract_features(x)
        # Pooling and final linear layer
        out1 = self.model._avg_pooling(out1)
        out1 = out1.flatten(start_dim=1)
        out1 = self.model._dropout(out1)
        out1 = self.classifier_layer(out1)
        out1 = self.fc(out1)

        return out1

class EfficientNet_b6(nn.Module):
    def __init__(self):
        super(EfficientNet_b6, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b6', dropout_rate=0.4, drop_connect_rate=0.2)
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.classifier_layer = nn.Sequential(
            nn.Linear(1280 , 512),
            nn.ReLU(512),
            nn.Dropout(0.4),
            nn.Linear(512 , 16)
        )

    def forward(self, x):

        out1 = self.model.extract_features(x)
        # Pooling and final linear layer
        out1 = self.model._avg_pooling(out1)
        out1 = out1.flatten(start_dim=1)
        out1 = self.model._dropout(out1)
        out1 = self.classifier_layer(out1)
        out1 = F.normalize(out1)

        return out1

class EfficientNet_V2_S(nn.Module):
    def __init__(self):
        super(EfficientNet_V2_S, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b0', pretrained=True, num_classes = 0)
        #self.model = EfficientNet.from_pretrained('efficientnet-b6', dropout_rate=0.4, drop_connect_rate=0.2)
        for param in self.model.parameters():
            param.requires_grad = True
        self.embedding = nn.Sequential(
            nn.Linear(1280 , 512),
            nn.ReLU(),
            nn.Dropout(0.2)        )
        self.dropout_layer = nn.Sequential(
            nn.Dropout(0.2)
        )
        self.final_fc = nn.Sequential(
            nn.Linear(512 , 16)
        )
        torch.nn.init.xavier_uniform_(self.embedding[0].weight)
        torch.nn.init.xavier_uniform_(self.final_fc[0].weight)
        #torch.nn.init.xavier_uniform_(self.classifier_layer[0].weight)
        #torch.nn.init.xavier_uniform_(self.classifier_layer[0].weight)


    def forward(self, x):
        """
        out1 = self.model.extract_features(x)
        # Pooling and final linear layer
        out1 = self.model._avg_pooling(out1)
        out1 = out1.flatten(start_dim=1)
        out1 = self.model._dropout(out1)
        out1 = self.classifier_layer(out1)
        out1 = F.normalize(out1)
        """
        out1 = self.model(x)
        out1 = self.dropout_layer(out1)
        out1 = self.embedding(out1)
        out1 = self.final_fc(out1)
        #out1 = F.normalize(out1)

        return out1



class EfficientNet_b0_baseline(nn.Module):
    def __init__(self):
        super(EfficientNet_b0_baseline, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')

        for param in self.model.parameters():
            param.requires_grad = True
        
        self.classifier_layer = nn.Sequential(
            nn.Linear(1280 , 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 , 16),
            nn.Linear(16 , 16),
            nn.Linear(16 , 16),
            nn.Linear(16 , 7)

        )
        torch.nn.init.xavier_uniform_(self.fc[0].weight)
        torch.nn.init.xavier_uniform_(self.fc[1].weight)
        torch.nn.init.xavier_uniform_(self.fc[2].weight)
        torch.nn.init.xavier_uniform_(self.fc[3].weight)


    def forward(self, x):

        out1 = self.model.extract_features(x)
        # Pooling and final linear layer
        out1 = self.model._avg_pooling(out1)
        out1 = out1.flatten(start_dim=1)
        out1 = self.model._dropout(out1)
        out1 = self.classifier_layer(out1)
        out1 = self.fc(out1)

        return out1

class EfficientNet_b0_Pretrained(nn.Module):
    def __init__(self, model):
        super(EfficientNet_b0_Pretrained, self).__init__()
        self.pretrained_model = model 
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(16 , 16),
            nn.ReLU(),
            nn.Linear(16 , 16),
            nn.ReLU(),
            nn.Linear(16 , 7)
             #            nn.Linear(256 , 104)
        )

        torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        torch.nn.init.xavier_uniform_(self.classifier[2].weight)
        torch.nn.init.xavier_uniform_(self.classifier[4].weight)

    def forward(self, x1):
        out1 = self.pretrained_model(x1)
        #out1 = self.pretrained_model.model.extract_features(x1)
        out1 = self.classifier(out1)
        return out1



# for test
if __name__ == '__main__':
    #model = EfficientNet_b0_baseline()
    model_1 = EfficientNet_b0()
    model = EfficientNet_b0_Pretrained(model_1)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

