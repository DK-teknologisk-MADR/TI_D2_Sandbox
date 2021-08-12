import cv2
import torch
import sys, os

import torchvision.transforms

import pytorch_ML.networks
import pytorch_ML.validators
print(torch.cuda.memory_summary())

import scipy
from filet.mask_discriminator.hyperopt import Mask_Hyperopt
import torch.nn as nn
from filet.mask_discriminator.mask_data_loader import rm_dead_data_and_get_ious, Filet_Seg_Dataset,Filet_Seg_Dataset_Box
from detectron2_ML.data_utils import get_file_pairs
import numpy as np
import torch.optim as optim
from pytorch_ML.validators import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau , ExponentialLR
from pytorch_ML.networks import IOU_Discriminator
from torchvision.transforms import Normalize
from filet.mask_discriminator.iou_transformations import get_r_mean_std,normalize_ious,rectify_ious

import os
img_dir="/pers_files/Combined_final/Filet/train"
data_dir = '/pers_files/mask_data_raw'
train_split = "train"
model_path = '/pers_files/mask_models_pad_mask35'
val_split = "val"

#os.makedirs(model_path)
file_pairs_train = get_file_pairs(data_dir,train_split)
file_pairs_val = get_file_pairs(data_dir,val_split)
data , iou_dict = rm_dead_data_and_get_ious(data_dir,train_split,file_pairs_train)
data_val , iou_dict_val = rm_dead_data_and_get_ious(data_dir,val_split,file_pairs_val)
iou_arr = np.array(list(iou_dict.values()))
#move to script in the end:
#-----------------------------

#r_mean,r_std = get_r_mean_std(iou_arr)
#iou_normed = rectify_ious(iou_arr)
#-----------------------------
#ty = [Normalize( mean=iou_arr.mean(), std=iou_arr.std())]
#tx = [Normalize( mean=[0.485, 0.456, 0.406,0], std=[0.229, 0.224, 0.225,1])]

dt = Filet_Seg_Dataset_Box(mask_dir=data_dir, img_dir=img_dir,split=train_split,trs_x= [] , trs_y=[rectify_ious])
dt_val = Filet_Seg_Dataset_Box(mask_dir=data_dir, img_dir=img_dir,split=val_split,trs_x= [] , trs_y=[rectify_ious])


# for i in range(500):
#     img,val = dt[i]
#     check[i] = img
# check.mean(axis=(0,2,3))
# check.std(axis=(0,2,3))

#TO SHOW:
#img = img.permute(1,2,0).numpy()
#cv2.imshow("window",img)
#cv2.imshow("window2",255*img[:,:,3])
#cv2.waitKey()
#cv2.destroyAllWindows()


from torchvision.transforms import ToTensor

import time


ious = iou_arr #CHANGE TO DT
#st_inds = np.argsort(ious)
#qs = np.arange(0.04,1.0,0.04)[:6]
quants =np.array([0.84,0.95]) #quants = np.quantile(iou_arr,qs)
#ious.std()
#np.hstack([quants])
intervals = np.searchsorted(quants, ious)
nrs = np.bincount(intervals)
weights_pre = nrs[0] / nrs
weights_pre = weights_pre * (1 / min(weights_pre))
weights_pre[1] = 0.01 #AVOID THOSE WE ARE IN DOUBT OF
print("loader will get weights",weights_pre, "for data between points",quants)
weights = weights_pre[intervals]
#model_path = "~/Pyscripts/test_folder"
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

output_dir =os.path.join(model_path,"classi_extreme_net")
os.makedirs(output_dir,exist_ok=False)
hyper = Mask_Hyperopt(base_lr=0.0001,base_path=model_path,max_iter = 150000,iter_chunk_size = 200,dt= dt,output_dir=output_dir,val_nr=1000, bs = 3,base_params= base_params,dt_val = dt_val,eval_period = 200,dt_wts = weights,fun_val=f1_score)
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
