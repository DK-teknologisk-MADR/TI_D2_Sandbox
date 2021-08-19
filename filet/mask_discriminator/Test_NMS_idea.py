import torch
import sys, os

import torchvision.transforms
import cv2
import pytorch_ML.networks
import pytorch_ML.validators
print(torch.cuda.memory_summary())
from torchvision.transforms.functional import affine
import scipy
from torch.utils.data.dataset import Subset
from filet.mask_discriminator.hyperopt import Mask_Hyperopt
import torch.nn as nn
from filet.mask_discriminator.mask_data_loader import rm_dead_data_and_get_ious, Filet_Seg_Dataset,Filet_Seg_Dataset_Box
from detectron2_ML.data_utils import get_file_pairs
import numpy as np
import torch.optim as optim
from pytorch_ML.validators import f1_score,f1_score_neg,mcc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau , ExponentialLR
from pytorch_ML.networks import IOU_Discriminator
from torchvision.transforms import Normalize
from detectron2_ML.predictors import ModelTester
from filet.mask_discriminator.iou_transformations import get_r_mean_std,normalize_ious,rectify_ious,Rectifier
import os
import detectron2.data.transforms as Tr
import torchvision.transforms as T
seg_model_dir ="/pers_files/Combined_final/Filet/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_4_output"
img_dir="/pers_files/Combined_final/Filet"
data_dir = '/pers_files/mask_data_raw_TV/'
train_split = "train"
model_path = '/pers_files/mask_models_pad_mask35_TV'
val_split = "val"
seg_model_fp = os.path.join(seg_model_dir,'best_model.pth')
seg_model_cfg_fp = os.path.join(seg_model_dir,'cfg.yaml')
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

dt = Filet_Seg_Dataset_Box(mask_dir=data_dir, img_dir=img_dir,split=train_split,trs_x= [] , trs_y=[Rectifier(.94,.94)])
dt_val = Filet_Seg_Dataset_Box(mask_dir=data_dir, img_dir=img_dir,split=val_split,trs_x= [] , trs_y=[Rectifier(.94,.94)])

files_to_keep = [(i,dt_val.get_files(i)) for i in range(len(dt_val)) if (dt_val.get_raw_y(i)<0.7)]
#
# #TO SHOW:
tester = ModelTester(cfg_fp=seg_model_cfg_fp,chk_fp=seg_model_fp)
t1min = Tr.RandomContrast(intensity_min=0.75,intensity_max=0.85)
t2min = Tr.RandomSaturation(intensity_min=0.75,intensity_max=0.85)
t3min = Tr.RandomLighting(0.8)
t1max = Tr.RandomContrast(intensity_min=1.15,intensity_max=1.25)
t2max = Tr.RandomSaturation(intensity_min=1.15,intensity_max=1.25)
t3max = Tr.RandomLighting(1.2)
t4 = Tr.RandomFlip(horizontal=True)
t5 = Tr.RandomRotation([-12,12],expand = False)
#augs = Tr.AugmentationList([t1,t2,t3,t4,t5])
augsmin = Tr.AugmentationList([t1min,t2min,t3min])
augsmax = Tr.AugmentationList([t1max,t2max,t3max])


def output_to_imshow_format(masks,index):
    return 255 * masks[index].to('cpu').numpy().astype('uint8')

#get_mask_votes(masks_ref,masks_aug,0.9)
def compute_ious(ref_mask,mask_batch,out):
    ref_mask_cps = ref_mask.expand_as(mask_batch)
    num =  torch.logical_and(ref_mask_cps,mask_batch).sum(axis=(1,2))
    den = torch.logical_or(ref_mask_cps,mask_batch).sum(axis=(1,2))
    if out is None:
        return torch.true_divide(input=num,other=den)
    else:
        torch.true_divide(input=num, other=den,out=out)

def get_mask_votes(masks_ref,masks_aug,th):
    masks_ref_nr = len(masks_ref)
    masks_aug_nr = len(masks_aug)
    ious = [torch.zeros(masks_aug_nr,device='cuda:0',dtype=torch.float) for i in range(masks_ref_nr)]
    for i in range(masks_ref_nr):
        compute_ious(masks_ref[i],masks_aug,out=ious[i])
    good_masks = [iou > th for iou in ious]
    return good_masks


def get_votes_from_good_masks(good_masks):
    votes_ls = [bool_mask.nonzero().flatten() for bool_mask in good_masks]
    return votes_ls

def show_result(masks_ref,masks_aug,vote_ls):
    for ref_mask_id in range(len(masks_ref)):
        voters = vote_ls[ref_mask_id]
        print(voters)
        img_cv = output_to_imshow_format(masks_ref, ref_mask_id)
        cv2.imshow(f"mask_orig{ref_mask_id}", img_cv)
        for voter in voters:
            img_cv = output_to_imshow_format(masks_aug,voter)
            cv2.imshow(f"mask{voter}",img_cv)
        cv2.waitKey()
        cv2.destroyAllWindows()


def post_process_nms(masks_ref,masks_aug,th):
    good_masks = get_mask_votes(masks_ref,masks_aug,th)
    votes = get_votes_from_good_masks(masks_ref,masks_aug,good_masks)

for img_id in range(40,50):
    img_fp = os.path.join(img_dir,files_to_keep[img_id][1][1])
    img = cv2.imread(img_fp)

    img_trs = []
    img = tester.predictor.aug.get_transform(img).apply_image(img)
    inp = Tr.AugInput(img)
    img_ts = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    img_trs.append({'image': img_ts})
    for i in range(4):
        if i<2:
            y = augsmin(inp)
        else:
            y = augsmax(inp)
        img_tr = y.apply_image(img)
        img_tr = torch.as_tensor(img_tr.astype("float32").transpose(2, 0, 1))
        if i % 2 == 0:
            img_tr=affine(img_tr,angle=0,translate = (61,43),scale=1.0,shear = 0)
        #    img_trs.append({'image' : img_tr , 'height ' : 1024, "width" : 1024})
        img_trs.append({'image' : img_tr})
    x = tester.predictor.model(img_trs)

    #mask_ls = [255*x[i]['instances'].pred_masks[0].to('cpu').numpy().astype('uint8') for i in range(5)]

    #for i in range(5):
    #    cv2.imshow(f"mask{i}",mask_ls[i])
    #cv2.waitKey()

    masks_ref = x[0]['instances'].pred_masks
    masks_aug = [ x[i]['instances'].pred_masks for i in range(1,len(x))]
    masks_aug = [affine(mask,angle=0,translate = (-61,-43),scale=1.0,shear = 0) if i % 2 == 0 else mask for i,mask in enumerate(masks_aug)]
    masks_aug = torch.cat(masks_aug,dim=0)


    good_masks = get_mask_votes(masks_ref,masks_aug,th=0.92)
    vote_ls = get_votes_from_good_masks(good_masks)
    show_result(masks_ref,masks_aug,vote_ls)