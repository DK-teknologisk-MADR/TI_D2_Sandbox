import torch
import os
import sys, os
sys.path.append('/pyscripts/pytorch_ML')
import validators
import pytorch_ML.networks
import pytorch_ML.validators
print(torch.cuda.memory_summary())
import scipy
import pytorch_ML.hyperopt as md
import torch.nn as nn
from filet_train.mask_discriminator.mask_data_loader import rm_dead_data_and_get_ious , get_file_pairs, Filet_Seg_Dataset
import numpy as np
import torch.optim as optim
from pytorch_ML.validators import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau , ExponentialLR
from pytorch_ML.networks import IOU_Discriminator
from torchvision.transforms import Normalize
import os
data_dir = '/pers_files/mask_pad_data19_centralized'
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
def get_r_mean_std(iou_arr):
    r_vals = np.where(iou_arr>0,iou_arr,0.01)
    r_vals = scipy.special.logit(r_vals)
    means , stds = r_vals.mean() , r_vals.std()
    return means, stds
r_mean,r_std = get_r_mean_std(iou_arr)


def normalize_ious(iou_arr):
    r_vals = np.where(iou_arr>0,iou_arr,0.01)
    r_vals = scipy.special.logit(r_vals)
    normed_r = (r_vals -r_mean) / r_std
    iou_normed = scipy.special.expit(normed_r)
    return iou_normed


def rectify_ious(iou_arr):
    iou_normed = np.where(iou_arr>0.95,1,iou_arr)
    iou_normed = np.where(iou_normed<0.90,0,iou_normed)
    iou_normed = np.where(np.logical_and(iou_normed>0.90, iou_normed < 0.95) ,iou_normed * 20 - 18,iou_normed)
    return iou_normed
#iou_normed = rectify_ious(iou_arr)
#-----------------------------
#ty = [Normalize( mean=iou_arr.mean(), std=iou_arr.std())]
tx = [Normalize( mean=[0.485, 0.456, 0.406,0], std=[0.229, 0.224, 0.225,1])]


dt = Filet_Seg_Dataset(data,iou_dict,data_dir,train_split,trs_x= tx , trs_y_bef=[rectify_ious],mask_only=False)
dt_val = Filet_Seg_Dataset(data_val,iou_dict_val,data_dir,val_split,trs_x= tx,trs_y_bef=[rectify_ious],mask_only=False)
dt[4]

ious = iou_arr #CHANGE TO DT
#st_inds = np.argsort(ious)
#qs = np.arange(0.04,1.0,0.04)[:6]
quants =np.array([0.9,0.95]) #quants = np.quantile(iou_arr,qs)
#ious.std()
#np.hstack([quants])
intervals = np.searchsorted(quants, ious)
nrs = np.bincount(intervals)
weights_pre = nrs[0] / nrs
weights_pre = weights_pre * (1 / min(weights_pre))
print("loader will get weights",weights_pre, "for data between points",quants)
weights = weights_pre[intervals]
model_path = "~/Pyscripts/test_folder"
base_params = {
    "optimizer": {
        'nesterov' : True,
    },
    "scheduler" : {},
    "optimizer_cls": optim.SGD,
    "scheduler_cls": ExponentialLR,
    "loss_cls": nn.BCEWithLogitsLoss,
    "net_cls": pytorch_ML.networks.IOU_Discriminator_01,
    "net": {'device': 'cuda'}
}
model_path = os.path.join("/pers_files/mask_models_pad_mask19")
os.makedirs(os.path.join(model_path,"classi_net"),exist_ok=True)
hyper = md.Hyperopt(base_path=model_path,max_iter = 250000,iter_chunk_size = 100,dt= dt,output_dir=os.path.join(model_path,"classi_net"),val_nr=None, bs = 3,base_params= base_params,dt_val = dt_val,eval_period = 180,dt_wts = None,fun_val=f1_score)
hyper.hyperopt()
#hyper = md.Hyperopt(model_path,max_iter = 250000,iter_chunk_size = 100,dt= dt,model_cls= IOU_Discriminator,optimizer_cls= optim.SGD,scheduler_cls= ReduceLROnPlateau,loss_cls= nn.BCEWithLogitsLoss,output_dir=model_path, bs = 3,base_params= base_params,dt_val = dt_val,eval_period = 180,dt_wts = weights)

#net = IOU_Discriminator(device='cuda:1')
#optimizer = optim.SGD(net.parameters(),lr = 0.05,momentum=0.23,nesterov= True)
#scheduler = ReduceLROnPlateau(optimizer,'min',0.23,patience = 5,verbose=True)
#loss_fun = nn.BCEWithLogitsLoss()
#fun_val = nn.BCEWithLogitsLoss(reduction='sum')

#trainer = md.Trainer_Save_Best(net=net,dt=dt, optimizer = optimizer, scheduler = scheduler, loss_fun =loss_fun , max_iter=5000, output_dir ='/pers_files/mask_models_pad_mask19/model49/trained_norm_y', eval_period=150, print_period=50,bs=3,
#                               dt_val=dt_val, dt_wts=weights, fun_val=fun_val, val_nr=None, add_max_iter_to_loaded=False)
#trainer.train()
