import torch

print(torch.cuda.memory_summary())

import filet_train.pytorch_ML.mask_discriminator as md
import torch.nn as nn
from filet_train.mask_discriminator.mask_data_loader import rm_dead_data_and_get_ious , get_file_pairs, Filet_Seg_Dataset
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from filet_train.pytorch_ML.networks import IOU_Discriminator
from torchvision.transforms import Normalize

device = 'cuda:0'
data_dir = '/pers_files/mask_pad_data'
train_split = "train"
model_path = '/pers_files/mask_models_pad_mask19'
val_split = "val"

file_pairs_train = get_file_pairs(data_dir,train_split)
file_pairs_val = get_file_pairs(data_dir,val_split)
data , iou_dict = rm_dead_data_and_get_ious(data_dir,train_split,file_pairs_train)
data_val , iou_dict_val = rm_dead_data_and_get_ious(data_dir,val_split,file_pairs_val)
iou_arr = np.array(list(iou_dict.values()))

#move to script in the end:
#-----------------------------

#-----------------------------

tx = [Normalize( mean=[0.485, 0.456, 0.406,0.425], std=[0.229, 0.224, 0.225,0.226])]
dt = Filet_Seg_Dataset(data,iou_dict,data_dir,train_split,trs_x= tx , trs_y_bef=[],mask_only=False)
dt_val = Filet_Seg_Dataset(data_val,iou_dict_val,data_dir,val_split,trs_x= tx,trs_y_bef=[],mask_only=False)
ious = iou_arr
st_inds = np.argsort(ious)
qs = np.arange(0.04,1.0,0.04)[:6]
quants = np.quantile(ious,qs)
ious.std()
np.hstack([quants])
intervals = np.searchsorted(quants, ious)
nrs = np.bincount(intervals)
weights_pre = nrs[0] / nrs

weights_pre = weights_pre * (1 / min(weights_pre))
print("loader will get weights",weights_pre, "for data between points",quants)
weights = weights_pre[intervals]

base_params = {
    "optimizer": {
        'nesterov' : True,
    },
    "scheduler" : {
        'mode' : 'min',
        'patience' : 5
    }
}

#hyper = md.Hyperopt(model_path,max_iter = 250000,iter_chunk_size = 100,dt= dt,model_cls= IOU_Discriminator,optimizer_cls= optim.SGD,scheduler_cls= ReduceLROnPlateau,loss_cls= nn.BCEWithLogitsLoss,output_dir=model_path, bs = 3,base_params= base_params,dt_val = dt_val,eval_period = 180,dt_wts = weights)

net = IOU_Discriminator(device=device)
optimizer = optim.SGD(net.parameters(),lr = 0.05,momentum=0.23,nesterov= True)
scheduler = ReduceLROnPlateau(optimizer,'min',0.23,patience = 4,verbose=True)
loss_fun = nn.BCEWithLogitsLoss()
fun_val = nn.BCEWithLogitsLoss(reduction='sum')

trainer = md.Trainer_Save_Best(net=net,dt=dt, optimizer = optimizer, scheduler = scheduler, loss_fun =loss_fun , max_iter=3000, output_dir ='/pers_files/mask_models_pad_mask19/model49/trained', eval_period=150, print_period=50,bs=3,
                               dt_val=dt_val, dt_wts=weights, fun_val=fun_val, val_nr=None, add_max_iter_to_loaded=False)
trainer.train()


