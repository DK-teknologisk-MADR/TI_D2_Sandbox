import cv2
import torch
import os
import sys, os
from filet.mask_discriminator.model_tester_mask import Model_Tester_Mask
import filet.mask_discriminator.mask_data_pairs_script
from pytorch_ML.networks import IOU_Discriminator_01
sys.path.append('/pyscripts/pytorch_ML')
import validators
import matplotlib.pyplot as plt
import pytorch_ML.networks
import pytorch_ML.validators
print(torch.cuda.memory_summary())
import scipy
import pytorch_ML.hyperopt as md
import torch.nn as nn
from filet.mask_discriminator.mask_data_loader import rm_dead_data_and_get_ious , get_file_pairs, Filet_Seg_Dataset , get_loader
import numpy as np

from filet.mask_discriminator.iou_transformations import normalize_ious,get_r_mean_std,rectify_ious
import torch.optim as optim
from pytorch_ML.validators import f1_score,prec_rec_scores
from torch.optim.lr_scheduler import ReduceLROnPlateau , ExponentialLR
from pytorch_ML.networks import IOU_Discriminator
from torchvision.transforms import Normalize
import os
img_dir="/pers_files/Combined_final/Filet/train"
data_dir = '/pers_files/mask_data_raw'
train_split = "train"
model_path = '/pers_files/mask_models_pad_mask35'
val_split = "val"
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

dt = Filet_Seg_Dataset(mask_dir=data_dir, img_dir=img_dir,split=train_split,trs_x= [] , trs_y=[rectify_ious])
dt_val = Filet_Seg_Dataset(mask_dir=data_dir, img_dir=img_dir,split=val_split,trs_x= [] , trs_y=[rectify_ious])

#load model
net = IOU_Discriminator_01(two_layer_head=False,device='cuda:1')
tester = Model_Tester_Mask(net=net,path=os.path.join(model_path,"classi_net1","model59","best_model.pth"),device='cuda:1')
loader = get_loader(dt_val,bs=10)
targets_ls,out_ls = [],[]
samples = []
fun=f1_score
aggregate_device = 'cuda:1'
for batch,targets in loader:
    batch = batch.to('cuda:1',non_blocking = True)
    target_batch = targets.to('cuda:1',non_blocking = True)
    out_batch = tester.get_evaluation(batch)
    targets_ls.append(target_batch)
    out_ls.append(out_batch)
    samples.append((batch[0].to('cpu').numpy(),targets[0].to('cpu').numpy(),out_batch[0].to('cpu').numpy()))
targets = torch.cat(targets_ls, 0)
outs = torch.cat(out_ls, 0).squeeze(1)  # dim i,j,: gives out if j= 0 and target if j = 1
scores = []
precs = []
recs = []
outs
ths = np.arange(0.5,1,0.05)
for th in ths:
    outs_th = torch.where(outs>th,1,0)
    score = fun(targets,outs_th)
    prec,rec = prec_rec_scores(targets,outs_th)
    score = score.to('cpu')
    scores.append(score)
    precs.append(prec)
    recs.append(rec)
    print("prec :",prec,"rec: ",rec,"f1: ",score)
plt.plot(ths,precs,label="precision")
plt.plot(ths,recs,label="rec")
plt.plot(ths,scores,label="score")
plt.legend(loc="upper left")
plt.axis([0.5, 1, 0, 1], 'equal')
plt.grid()

#samples_fixed = [(img.transpose(1,2,0),target,out[0]) for img,target,out in samples]

for img,target,out in samples_fixed:
    print("target",target,"out",out)
    cv2.namedWindow("meat")
    cv2.moveWindow("meat", 500, 500)
    cv2.imshow("meat", img[:, :, :3])
    cv2.namedWindow("mask")
    cv2.moveWindow("mask", 1500, 500)
    cv2.imshow("mask", img[:, :, 3])
    cv2.waitKey()
    cv2.destroyAllWindows()