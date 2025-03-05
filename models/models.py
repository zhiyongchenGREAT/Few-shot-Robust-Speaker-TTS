import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class classifier_spk_eresnet(nn.Module):
    def __init__(self, num_classes):
        super(classifier_spk_eresnet, self).__init__()
        self.num_classes = num_classes

        self.bn1 = nn.InstanceNorm1d(512)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(192, 512, bias=False)

        self.bn1_1 = nn.InstanceNorm1d(256)
        self.relu1_1 = nn.ReLU()
        self.fc1_1 = nn.Linear(512, 256, bias=False)

        self.bn1_2 = nn.InstanceNorm1d(256)
        self.relu1_2 = nn.ReLU()
        self.fc1_2 = nn.Linear(256, 256, bias=False)

        self.reducedim=nn.Linear(256,192,bias=False)
        self.fc2 = nn.Linear(192, num_classes, bias=False)

    def forward(self, x, return_feature=False):
        if len(x.shape)<2:
            x = x.unsqueeze(0)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        x = self.fc1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        x=self.reducedim(x)
        y = self.fc2(x)

        if return_feature:
            return x, y
        else:
            return y