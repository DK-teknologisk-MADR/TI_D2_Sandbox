import torch
from filet_train.mask_discriminator.Model_Tester import Model_Tester
from filet_train.mask_discriminator.train_script import IOU_Discriminator,dt_val
from filet_train.mask_discriminator.mask_data_loader import get_loader , rm_dead_data_and_get_ious , get_file_pairs, Filet_Seg_Dataset
from torchvision import transforms

import numpy as np
import os
from Model_Tester import Model_Tester

class Model_Tester_Mask(Model_Tester):
    def __init__(self,net,path,trs_x = [],device='cuda:0'):
        self.trs_x = trs_x
        self.device = device
        super().__init__(net=net,path_to_save_file=path,device=device)


    def get_evaluation(self,picture):
        '''
        assumes CxHxW picture
        '''
        torch.tensor(picture,device=self.device,requires_grad=False)
        if not self.trs_x == []:
            pic = self.trs_x(picture)
        res = self.net(picture)
        return res
#move to script in the end:
#-----------------------------
def normalize_ious(arr):
    x0 = 0.8
    x1 = 0.94
    arr = (arr - x0) * 1/(x1-x0)
    return np.where(arr<0,0,
             np.where(arr>1,1,arr))

#----------------

data_dir = '/pers_files/mask_data'
train_split = "train"
model_path = '/pers_files/mask_models_WithMask'
val_split = "val"

tx = [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]

file_pairs_train = get_file_pairs(data_dir,train_split)
file_pairs_val = get_file_pairs(data_dir,val_split)
data , iou_dict = rm_dead_data_and_get_ious(data_dir,train_split,file_pairs_train)
data_val , iou_dict_val = rm_dead_data_and_get_ious(data_dir,val_split,file_pairs_val)
#dt = Filet_Seg_Dataset(data,iou_dict,data_dir,train_split,trs_y_bef=[normalize_ious],mask_only=False)
dt_val = Filet_Seg_Dataset(data_val,iou_dict_val,data_dir,val_split,trs_x=[],trs_y_bef=[normalize_ious],mask_only=False)

dt_val[0]

net = IOU_Discriminator()
x = Model_Tester_Mask(net,f'{model_path}/model165/best_model.pth')
x(dt_val[0][0])