import pandas as pd
import torch
import pytorch_ML.networks
import pytorch_ML.validators
print(torch.cuda.memory_summary())
from torchvision.models import resnet101
from torch.utils.data.dataset import Subset
from filet.mask_discriminator.hyperopt import Mask_Hyperopt
import torch.nn as nn
from pytorch_ML.trainer import Trainer
from filet.mask_discriminator.mask_data_loader import rm_dead_data_and_get_ious, Filet_Seg_Dataset,Filet_Seg_Dataset_Box
from detectron2_ML.data_utils import get_file_pairs
import numpy as np
import torch.optim as optim
from torch.optim import SGD
from pytorch_ML.networks import IOU_Discriminator_01
from pytorch_ML.validators import f1_score,f1_score_neg,mcc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau , ExponentialLR
from filet.mask_discriminator.transformations import PreProcessor_Crop_n_Resize_Box
from filet.mask_discriminator.iou_transformations import get_r_mean_std,normalize_ious,rectify_ious,Rectifier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
import os
from detectron2_ML.pruners import SHA
img_dir="/pers_files/Combined_final/Filet"
data_dir = '/pers_files/mask_data_raw_TV/'
train_split = "train"
model_base_path = '/pers_files/mask_models_pad_mask_hyper'
val_split = "val"
model_dir = os.path.join(model_base_path,"classi_net_TV_rect_balanced_mcc_score_fixedf4/model86")
output_dir = os.path.join(model_dir,'trained_model')
os.makedirs(output_dir)
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
#base_prep = PreProcessor_Crop_n_Resize_Box(resize_dims=(393,618),pad=35,mean=[0, 0, 0, 0.0000],std=[1, 1, 1, 1])  #not used
#dt_to_compute_mean = Filet_Seg_Dataset_Box(mask_dir=data_dir, img_dir=img_dir,split=train_split,preprocessor=base_prep , trs_y=[Rectifier(.94,.94)])
#imgs = []
#for i in range(400):
#    img,_ = dt_to_compute_mean[i]
#    imgs.append(img)
#imgs = torch.stack(imgs)
pad = 50
resize = [255,255]
prep = PreProcessor_Crop_n_Resize_Box(resize_dims=[255,255],pad=50,mean=[0.2010, 0.1944, 0.2488, 0.0000],std=[0.224, 0.224, 0.224, 1])
dt = Filet_Seg_Dataset_Box(mask_dir=data_dir, img_dir=img_dir,split=train_split,preprocessor=prep , trs_y=[Rectifier(.92,.92)])
dt_val = Filet_Seg_Dataset_Box(mask_dir=data_dir, img_dir=img_dir,split=val_split,preprocessor=prep , trs_y=[Rectifier(.92,.92)])

points_to_keep = [(i,dt.get_raw_y(i)) for i in range(len(dt)) if (dt.get_raw_y(i)<0.80 or dt.get_raw_y(i)>0.92)]
indices_to_keep = [x for x,y in points_to_keep]
ys_to_keep = np.array([y for x,y in points_to_keep])
dt = Subset(dt,indices_to_keep)
dt_val = Subset(dt_val,[i for i in range(len(dt_val)) if (dt_val.get_raw_y(i)<0.85 or dt_val.get_raw_y(i)>0.92) ])

#
# #TO SHOW:
# for i in range(len(dt)):
#     img,target = dt[i]
#     img = img.permute(1,2,0).numpy()
#     cv2.imshow("window",img)
#     cv2.imshow("window2",255*img[:,:,3])
#     cv2.waitKey()
#     cv2.destroyAllWindows()
# import time
#

quants =np.array([0.80]) #quants = np.quantile(iou_arr,qs)
#ious.std()
#np.hstack([quants])
intervals = np.searchsorted(quants, ys_to_keep)
nrs = np.bincount(intervals)
weights_pre = nrs[0] / nrs
weights_pre = weights_pre * (1 / min(weights_pre))
print("loader will get weights",weights_pre, "for data between points",quants)
weights = weights_pre[intervals]
two_layer_head=False
net = IOU_Discriminator_01(two_layer_head=False)
optimizer = SGD(net.parameters(),lr=0.02,momentum=0.218,nesterov=True)
scheduler = ReduceLROnPlateau(optimizer,mode='max',factor=0.8,patience=250*7,verbose=True,threshold = 0.01)
loss_fun = BCEWithLogitsLoss()
trainer = Trainer(dt=dt,net=net,optimizer=optimizer,scheduler=scheduler,loss_fun=loss_fun,max_iter = 1000000,output_dir=output_dir,eval_period=250,print_period=50,bs=4,dt_val=dt_val,fun_val=mcc_score,gpu_id=1)
trainer.load(os.path.join(model_dir,'best_model.pth'),skip=['scheduler'])
trainer.train()