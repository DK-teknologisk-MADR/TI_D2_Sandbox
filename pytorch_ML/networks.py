import torch
import torchvision.models
from torchvision.models import wide_resnet50_2,resnet50
import torch.nn as nn
model = wide_resnet50_2(False)


class FChead(nn.Module):
    def __init__(self,dims,device):
        super(FChead, self).__init__()
        assert len(dims) == 4
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(dims[0],dims[1])
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dims[1],dims[2])
        self.fcout = nn.Linear(dims[2],dims[3])
    def forward(self,x):
        x = self.relu(self.fc1(self.dropout1(x)))
        x = self.relu(self.fc2(self.dropout2(x)))
        x = self.fcout(self.dropout3(x))
        return x


class IOU_Discriminator_Only_Mask(nn.Module):
    def __init__(self,device):
        super(IOU_Discriminator_Only_Mask, self).__init__()
        self.model_wide_res = wide_resnet50_2(False)
        self.fchead = FChead(dims = [2048,400,400,1],device = device)
        self.model_wide_res.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.device = device
    def forward(self,x):
        x = self.model_wide_res.forward(x)
        x = self.fchead(x)
        return x


class IOU_Discriminator(nn.Module):
    def __init__(self,device = 'cuda:1'):
        super(IOU_Discriminator, self).__init__()
        self.device = device
        self.model_wide_res = wide_resnet50_2(True)
        self.model_wide_res.fc = nn.Identity()
        self.fchead = FChead(dims = [2048,400,400,1],device = device)
        weight = self.model_wide_res.conv1.weight.clone()
        self.model_wide_res.conv1 = nn.Conv2d(4, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        with torch.no_grad():
            self.model_wide_res.conv1.weight[:, :3] = weight
        self.model_wide_res.to(self.device)
        self.to(self.device)
    def forward(self,x):
        x = self.model_wide_res.forward(x)
        x = self.fchead(x)
        return x



class Backbone_And_Fc_Head(nn.Module):
    def __init__(self,backbone = resnet50,fc_dims = (1024,300,300,1), device = 'cuda:1'):
        self.device = device
        super(Backbone_And_Fc_Head, self).__init__()
        self.backbone = backbone(True)
        self.fchead = FChead(dims = fc_dims,device = device)
        self.backbone.to(self.device)
        self.to(self.device)
    def forward(self,x):
        x = self.backbone.forward(x)
        x = self.fchead(x)
        return x
