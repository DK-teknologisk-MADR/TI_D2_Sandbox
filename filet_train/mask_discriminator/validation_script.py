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

from iou_transformations import normalize_ious,get_r_mean_std,rectify_ious
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


dt_val = Filet_Seg_Dataset(data_val,iou_dict_val,data_dir,val_split,trs_x= tx,trs_y_bef=[normalize_ious],mask_only=False)
