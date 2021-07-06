import torch
from torchvision.models import wide_resnet50_2
import torch.nn as nn


class IOU_Discriminator(nn.Module):
    def __init__(self,device = 'cuda:1'):
        super(IOU_Discriminator, self).__init__()
        self.model_wide_res = wide_resnet50_2(True)
        weight = self.model_wide_res.conv1.weight.clone()
        self.model_wide_res.conv1 = nn.Conv2d(4, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model_wide_res.fc = nn.Linear(2048,400)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(400,400)
        self.dropout2 = nn.Dropout(0.25)
        self.fcout = nn.Linear(400,1)
        self.device = device
        with torch.no_grad():
            self.model_wide_res.conv1.weight[:, :3] = weight
        self.model_wide_res.to(self.device)
        self.to(self.device)
    def forward(self,x):
        x = self.model_wide_res.forward(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fcout(x)
        return x



class IOU_Discriminator_Only_Mask(nn.Module):
    def __init__(self,device):
        super(IOU_Discriminator_Only_Mask, self).__init__()
        self.model_wide_res = wide_resnet50_2(False)
        self.model_wide_res.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model_wide_res.fc = nn.Linear(2048,400)
        self.device = device
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(400,400)
        self.dropout2 = nn.Dropout(0.25)
        self.fcout = nn.Linear(400,1)
        self.model_wide_res.to(device)

    def forward(self,x):
        x = self.model_wide_res.forward(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fcout(x)
        return x
