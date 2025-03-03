import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()
        
        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()
        
        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()
        
        self.fc1 = nn.Linear(128*4*4, 2)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x, rf=False):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 128*4*4)
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)
        
        if rf:
            return x, y
        return y

def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class classifier32(nn.Module):
    def __init__(self, num_classes=10):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes, bias=False)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_feature=False):
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)
        if return_feature:
            return x, y
        else:
            return y

import os

def load_and_average_embeddings(dir_path, num_speakers=15, num_utterances=10):
    mat_enrolled = torch.zeros(num_speakers, 512)  # Assuming embeddings are of size 512
    for spk_id in range(num_speakers):
        spk_embeddings = []
        for utt_id in range(num_utterances):
            file_path = os.path.join(dir_path, str(spk_id), f"{utt_id}.npy")
            embedding = torch.load(file_path)
            embedding_tensor = torch.from_numpy(embedding)
            spk_embeddings.append(embedding_tensor)
        # Convert list of tensors to a tensor, then compute mean along the 0th dimension
        spk_embeddings_tensor = torch.stack(spk_embeddings)
        mat_enrolled[spk_id] = torch.mean(spk_embeddings_tensor, dim=0)
    return mat_enrolled

class classifier_spk(nn.Module):
    def __init__(self, num_classes):
        super(classifier_spk, self).__init__()
        self.num_classes = num_classes

        # Define batch normalization and ReLU activation
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(512, 512, bias=False)

        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.ReLU()
        self.fc1_1 = nn.Linear(512, 256, bias=False)

        self.bn1_2 = nn.BatchNorm1d(256)
        self.relu1_2 = nn.ReLU()
        self.fc1_2 = nn.Linear(256, 256, bias=False)

        # self.bn1_3 = nn.BatchNorm1d(256)
        # self.relu1_3 = nn.ReLU()
        # self.fc1_3 = nn.Linear(256, 256, bias=False)

        # self.bn1_4 = nn.BatchNorm1d(256)
        # self.relu1_4 = nn.ReLU()
        # self.fc1_4 = nn.Linear(256, 256, bias=False)

        # self.bn1_5 = nn.BatchNorm1d(512)
        # self.relu1_5 = nn.ReLU()
        # self.fc1_5 = nn.Linear(512, 512, bias=False)

        # Define the fully connected layers
        # self.bn2 = nn.BatchNorm1d(256)
        # self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes, bias=False)

        # mat_enrolled = load_and_average_embeddings('/nvme/zhiyong/avsp/vox1_t_train_emb')
        # self.fc2.weight = nn.Parameter(mat_enrolled)
        # print(self.fc2.weight.shape, mat_enrolled.shape)

    def forward(self, x, return_feature=False):
        # Flatten the input
        x = torch.flatten(x, 1)
        
        # Apply the first set of layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Apply the second set of layers
        x = self.fc1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)


        # Apply the third set of layers
        x = self.fc1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        # # Apply the fourth set of layers
        # x = self.fc1_3(x)
        # x = self.bn1_3(x)
        # x = self.relu1_3(x)

        # # Apply the fourth set of layers
        # x = self.fc1_4(x)
        # x = self.bn1_4(x)
        # x = self.relu1_4(x)

        # # Apply the fourth set of layers
        # x = self.fc1_5(x)
        # x = self.bn1_5(x)
        # x = self.relu1_5(x)

        # x = x
        # Apply the final fully connected layer to produce logits for classification
        y = self.fc2(x)

        # If return_feature is True, return both the feature and the output
        if return_feature:
            return x, y
        else:
            return y
        


import torch
import torch.nn as nn

class classifier_spk_frame(nn.Module):
    def __init__(self, num_classes=15):
        super(classifier_spk_frame, self).__init__()
        self.gru = nn.GRU(input_size=768, hidden_size=512, batch_first=True)
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, return_feature=False):
        x, _ = self.gru(x)  # GRU for combining features
        x = torch.mean(x, dim=1)  # Mean pooling over time
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        y = self.fc2(x)
        if return_feature:
            return x, y
        else:
            return y




class classifier_spk_eresnet(nn.Module):
    def __init__(self, num_classes):
        super(classifier_spk_eresnet, self).__init__()
        self.num_classes = num_classes

        # Define batch normalization and ReLU activation
        # self.bn1 = nn.BatchNorm1d(512)
        self.bn1 = nn.InstanceNorm1d(512)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(192, 512, bias=False)

        # self.bn1_1 = nn.BatchNorm1d(256)
        self.bn1_1 = nn.InstanceNorm1d(256)
        self.relu1_1 = nn.ReLU()
        self.fc1_1 = nn.Linear(512, 256, bias=False)

        # self.bn1_2 = nn.BatchNorm1d(256)
        self.bn1_2 = nn.InstanceNorm1d(256)
        self.relu1_2 = nn.ReLU()
        self.fc1_2 = nn.Linear(256, 256, bias=False)

        # self.fc2 = nn.Linear(256, num_classes, bias=False)
        self.reducedim=nn.Linear(256,192,bias=False)
        self.fc2 = nn.Linear(192, num_classes, bias=False)

    def forward(self, x, return_feature=False):
        # Flatten the input
        # print("forward x shape:",x.shape)
        if len(x.shape)<2:
            x=x.unsqueeze(0)
        x = torch.flatten(x, 1)
        
        # Apply the first set of layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Apply the second set of layers
        x = self.fc1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)


        # Apply the third set of layers
        x = self.fc1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        # Apply the final fully connected layer to produce logits for classification
        # print(x.shape)
        x=self.reducedim(x)
        y = self.fc2(x)
        # print(y.shape)
        # If return_feature is True, return both the feature and the output
        if return_feature:
            return x, y
        else:
            return y


