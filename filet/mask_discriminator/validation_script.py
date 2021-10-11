import cv2
import torch
import sys
from pytorch_ML.model_tester_mask import Model_Tester_Mask
from pytorch_ML.networks import IOU_Discriminator_01
sys.path.append('/pyscripts/pytorch_ML')
import matplotlib.pyplot as plt

print(torch.cuda.memory_summary())
from filet.mask_discriminator.mask_data_loader import rm_dead_data_and_get_ious , get_file_pairs, get_loader,Filet_Seg_Dataset_Box
import numpy as np

from filet.mask_discriminator.iou_transformations import Rectifier
from pytorch_ML.validators import f1_score_neg, prec_rec_spec_neqprec_scores
import os
img_dir="/pers_files/Combined_final/Filet"
data_dir = '/pers_files/mask_data_raw_TV/'
train_split = "train"
model_path = '/pers_files/mask_models_pad_mask35_TV/classi_net_TV_rect_balanced_mcc_score_fixed/model20'
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

dt = Filet_Seg_Dataset_Box(mask_dir=data_dir, img_dir=img_dir,split=train_split,trs_x= [] , trs_y=[Rectifier(0.90,0.90)])
dt_val = Filet_Seg_Dataset_Box(mask_dir=data_dir, img_dir=img_dir,split=val_split,trs_x= [] , trs_y=[Rectifier(0.90,0.90)])
img,target = dt[0]
img = img.numpy()
img = img.transpose(1,2,0)
np.sum(img[:,:,3]==1)
#load model
net = IOU_Discriminator_01(two_layer_head=True,device='cuda:1')
tester = Model_Tester_Mask(net=net,path=os.path.join(model_path,"best_model.pth"),device='cuda:1')
bs=10
loader = get_loader(dt_val,bs=bs,replacement=False)
targets_ls,out_ls = [],[]
samples = []
fun=f1_score_neg
aggregate_device = 'cuda:1'
for batch,targets in loader:
    batch = batch.to('cuda:1')
    target_batch = targets.to('cuda:1')
    out_batch = tester.get_evaluation(batch)
    targets_ls.append(target_batch)
    out_ls.append(out_batch)
    for i in range(3):
        samples.append((batch[i].to('cpu').numpy(),targets[i].to('cpu').numpy(),out_batch[i].to('cpu').numpy()))

outs = torch.cat(out_ls, 0).squeeze(1)  # dim i,j,: gives out if j= 0 and target if j = 1
targets = torch.cat(targets_ls, 0)  # dim i,j,: gives out if j= 0 and target if j = 1
scores = []
precs = []
recs = []
specs = []
neg_precs = []
ths = np.arange(0.15,1,0.05)
for th in ths:
    outs_th = torch.where(outs>th,1,0)
    score = fun(targets,outs_th)
    prec,rec,spec,neg_prec = prec_rec_spec_neqprec_scores(targets,outs_th)
    score = score.to('cpu')
    scores.append(score)
    precs.append(prec)
    recs.append(rec)
    specs.append(spec)
    neg_precs.append(neg_prec)
    print("prec :",prec,"rec: ",rec,"f1: ",score,)
plt.plot(ths,precs,label="precision")
plt.plot(ths,recs,label="recall")
plt.plot(ths,specs,label="specificity")
plt.plot(ths,neg_precs,label="negative_precision")
plt.plot(ths,scores,label="score")
plt.legend(loc="upper left")
plt.axis([0, 1, 0, 1], 'equal')
plt.xlabel("threshholds")
plt.ylabel("values")

plt.show()


th=0.5
samples_fixed = [(img.transpose(1,2,0),target,out[0]) for img,target,out in samples]
len(samples_fixed)
count = 0

for img,target,out in samples_fixed:
#    if out<0.65 and target == 0:
    print("target",target,"out",out,"area",img[:, :, 3].sum(axis=(0,1)))
    cv2.namedWindow("meat")
    cv2.moveWindow("meat", 500, 500)
    cv2.imshow("meat", img[:, :, :3])
    cv2.namedWindow("mask")
    cv2.moveWindow("mask", 1500, 500)
    cv2.imshow("mask", img[:, :, 3])
    cv2.waitKey()
    cv2.destroyAllWindows()