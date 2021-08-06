from collections import OrderedDict
import torch
import numpy as np
import torchvision.models
from torchvision.models import wide_resnet50_2,resnet50
from time import time
import torch.nn as nn

#model = wide_resnet50_2(False)

class FChead(nn.Module):
    def __init__(self,dims,dropout_ps = None ,activ_ls = None,device='cuda:0'):
        super().__init__()
        '''
        dims : list of length n+1 where dims[0] is input dim, dim[n] = output dim
        dropout_ps : dropout probs. 0 means it should not be added
        '''
        layer_dict = [nn.Linear(dims[i],dims[i+1]) for i,dim in enumerate(dims[:-1])]
        if dropout_ps is None:
            dropout_ps = np.zeros(len(dims) - 1)
        do_dict = [nn.Dropout(dropout_ps[i]) for i in range(len(dims) - 1) ]
        if activ_ls is None:
            activ_ls = [nn.ReLU() for _ in range(len(dims) - 2)]
            activ_ls.append(nn.Identity())
        nn_list = OrderedDict()
        for i,layer in enumerate(layer_dict):
            if dropout_ps[i]:
                nn_list['dp' + str(i)] = do_dict[i]
            nn_list['lin' + str(i)] = layer_dict[i]
            nn_list['activ' + str(i)] = activ_ls[i]
        self.nn_seq = nn.Sequential(nn_list)

    def forward(self,x):
        x = self.nn_seq.forward(x)

        return x

class IOU_Discriminator_Only_Mask(nn.Module):
    def __init__(self,device):
        super(IOU_Discriminator_Only_Mask, self).__init__()
        self.model_wide_res = wide_resnet50_2(False)
        self.fchead = FChead(dims = [2048,400,400,1],dropout_ps= [0.2,0.2,0.2] ,device = device)
        self.model_wide_res.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.device = device
    def forward(self,x):
        x = self.model_wide_res.forward(x)
        x = self.fchead(x)
        return x


class IOU_Discriminator(nn.Module):
    def __init__(self,device = 'cuda:0'):
        super(IOU_Discriminator, self).__init__()
        self.device = device
        self.model_wide_res = wide_resnet50_2(True)
        self.model_wide_res.fc = nn.Identity()
        self.fchead = FChead(dims = [2048,400,400,1],dropout_ps= [0.2,0.2,0.2],device = device)
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




class IOU_Discriminator_Sig_MSE(IOU_Discriminator):
    def __init__(self, device='cuda:0'):
        super(IOU_Discriminator_Sig_MSE, self).__init__(device = device)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = super(IOU_Discriminator_Sig_MSE, self).forward(x)
        x = self.sig(x)
        return x


class IOU_Discriminator_01(nn.Module):
    def __init__(self,backbone = None,two_layer_head = True, device='cuda:0'):
        '''
        needs to be a backbone whose first layer is "conv1" which we replaces to have 4 channels.
        Such as all resnet and wide_resnets
        '''
        if backbone is None:
            backbone = wide_resnet50_2(True)
        weight = backbone.conv1.weight.clone()
        backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        backbone.fc = nn.Identity()
        with torch.no_grad():
            backbone.conv1.weight[:, :3] = weight
        if two_layer_head:
            fcHead = FChead(dims=[2048, 1024,1],dropout_ps = [0,0.1], device=device)
        else:
            fcHead = FChead(dims=[2048,1], device=device)

        super(IOU_Discriminator_01, self).__init__()
        self.model = Backbone_And_Fc_Head(backbone, fcHead, device)
        self.sigmoid =nn.Sigmoid()
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = self.model.forward(x)
        if not self.training: #if eval mode
            x = self.sigmoid(x)
        return x


class Backbone_And_Fc_Head(nn.Module):
    def __init__(self,backbone = None,fcHead = None, device = 'cuda:0'):
        self.device = device
        super(Backbone_And_Fc_Head, self).__init__()
        if backbone is None:
            self.backbone = resnet50(False)
        else:
            self.backbone = backbone
        self.backbone.fc = nn.Identity()
        self.fcHead = fcHead
        self.backbone.to(self.device)
        self.fcHead.to(self.device)
        self.to(self.device)

    def forward(self,x):
        x = self.backbone.forward(x)
        x = self.fcHead.forward(x)
        return x

class WRN_Regressor(Backbone_And_Fc_Head):
    def __init__(self,output_dim = 1,last_activ = None,pretrained  = True, device = 'cuda:0'):
        '''
        Pure wide resnet 50 with last output changed to 2048-> output_dim and last_activ activation function.
        WARNING: Meant to be used with either
        '''
        if last_activ is None:
            last_activ = nn.Identity()
        super().__init__(backbone = wide_resnet50_2(True), fcHead = FChead(dims = [2048,output_dim],activ_ls=[last_activ],device = device))
        self.backbone.fc = nn.Identity()
        self.device = device


        self.to(self.device)
    def forward(self,x):
        return super(WRN_Regressor,self).forward(x)


def try_script_model(model,sample_shape,device = 'cuda:0',reps = 20,tolerance = 1e-7):
    model.eval()
    model_jit  = torch.jit.script(model)
    results = []
    burn_in = reps//2
    timings_raw = np.zeros(reps-burn_in)
    timings_jit = np.zeros(reps-burn_in)


    for i in range(reps):
        with torch.no_grad():
            x = torch.randn(sample_shape,requires_grad=False,device=device)
            timeBegin = time()
            y_raw = model(x)
            time_end = time()-timeBegin
            if i>=burn_in:
              timings_raw[i-burn_in] = time_end
            timeBegin = time()
            y_jit = model_jit(x)
            time_end = time() - timeBegin
            if i>=burn_in:
                timings_jit[i-burn_in] = time_end
            results.append(torch.abs(y_raw-y_jit))
        result_ts = torch.vstack(results)
        result_ts = result_ts.mean(axis=0)
    if torch.all( result_ts< tolerance):
        print("try_script_model:: succesfully  jitted  model")
        print(f"try_script_model::average timings based on {reps-burn_in} trials: raw : {timings_raw.mean():3f}, jit : {timings_jit.mean():3f}")
        return model_jit , True
    else:
        print("try_script_model:: failed jit model")
        return model , False




#TODO::make test that torch.jit.scripts everything.


#try_script_model(tester,(3,4,300,300))
