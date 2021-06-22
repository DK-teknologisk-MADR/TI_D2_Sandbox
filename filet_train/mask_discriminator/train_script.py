import mask_discriminator as md
import torch.nn as nn
from torchvision.models import wide_resnet50_2
from mask_data_loader import get_loader , rm_dead_data_and_get_ious , get_file_pairs, Filet_Seg_Dataset
from torchvision.transforms import Normalize
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
import torch
class IOU_Discriminator(nn.Module):
    def __init__(self):
        super(IOU_Discriminator, self).__init__()
        self.model_wide_res = wide_resnet50_2(True)
        weight = self.model_wide_res.conv1.weight.clone()
        self.model_wide_res.conv1 = nn.Conv2d(4, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model_wide_res.fc = nn.Linear(2048,400)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(400,400)
        self.dropout2 = nn.Dropout(0.25)
        self.fcout = nn.Linear(400,1)
        with torch.no_grad():
            self.model_wide_res.conv1.weight[:, :3] = weight
        self.model_wide_res.to(md.compute_device)

    def forward(self,x):
        x = self.model_wide_res.forward(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fcout(x)
        return x

class IOU_Discriminator_Only_Mask(nn.Module):
    def __init__(self):
        super(IOU_Discriminator_Only_Mask, self).__init__()
        self.model_wide_res = wide_resnet50_2(False)
        self.model_wide_res.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model_wide_res.fc = nn.Linear(2048,400)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(400,400)
        self.dropout2 = nn.Dropout(0.25)
        self.fcout = nn.Linear(400,1)
        self.model_wide_res.to(md.compute_device)

    def forward(self,x):
        x = self.model_wide_res.forward(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fcout(x)
        return x



data_dir = '/pers_files/mask_data'
train_split = "train"
model_path = '/pers_files/mask_models_Only_Mask'
val_split = "val"

file_pairs_train = get_file_pairs(data_dir,train_split)
file_pairs_val = get_file_pairs(data_dir,val_split)
data , iou_dict = rm_dead_data_and_get_ious(data_dir,train_split,file_pairs_train)
data_val , iou_dict_val = rm_dead_data_and_get_ious(data_dir,val_split,file_pairs_val)

#move to script in the end:
#-----------------------------
def normalize_ious(arr):
    x0 = 0.8
    x1 = 0.94
    arr = (arr - x0) * 1/(x1-x0)
    return np.where(arr<0,0,
             np.where(arr>1,1,arr))

#-----------------------------

dt = Filet_Seg_Dataset(data,iou_dict,data_dir,train_split,trs_y_bef=[normalize_ious],mask_only=True)
dt_val = Filet_Seg_Dataset(data_val,iou_dict_val,data_dir,val_split,trs_y_bef=[normalize_ious],mask_only=True)
ious = dt_val.ious #CHANGE TO DT
st_inds = np.argsort(ious)
qs = np.array([0.80, 0.94])
intervals = np.searchsorted(qs, ious)
nrs = np.bincount(intervals)
weights_pre = nrs[0] / nrs
weights_pre = weights_pre * (1 / min(weights_pre))
weights = weights_pre[intervals]
base_params = {
    "optimizer": {
        'nesterov' : True,
    },
    "scheduler" : {
        'mode' : 'min',
        'patience' : 10
    }
}

hyper = md.Hyperopt(model_path,max_iter = 350000,iter_chunk_size = 100,dt= dt,model_cls= IOU_Discriminator_Only_Mask,optimizer_cls= optim.SGD,scheduler_cls= ReduceLROnPlateau,loss_cls= nn.BCEWithLogitsLoss,output_dir=model_path, bs = 4,base_params= base_params,dt_val = dt_val,eval_period = 250,dt_wts = weights)

hyper.hyperopt()
