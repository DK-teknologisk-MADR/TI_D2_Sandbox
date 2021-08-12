import numpy as np
import torch
from torchvision.transforms import Compose
import cv2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import json
import os
import copy
from detectron2_ML.data_utils import get_file_pairs
from filet.mask_discriminator.transformations import PreProcessor,PreProcessor_Box_Crop

#import matplotlib
#matplotlib.use('TkAgg')



def rm_dead_data_and_get_ious(data_dir,split,file_pairs = None):
    if file_pairs is None:
        file_pairs = get_file_pairs(data_dir,split)
    iou_dict= {}
    data = file_pairs.copy()
    for front,files in file_pairs.items():
        for file in files:
            if file.endswith("mask.jpg"):
                mask_file = os.path.join(data_dir,split,file)
            elif file.endswith(".jpg"):
                jpg_file =  os.path.join(data_dir,split,file)
            elif file.endswith(".json"):
                json_file = os.path.join(data_dir,split,file)
            else:
                print("warning, cant recognize file",file)
    #    mask = cv2.imread(mask_file)
     #   filet = cv2.imread(jpg_file)
        with open(os.path.join(data_dir,split,json_file)) as fp:
            json_dict = json.load(fp)
            iou = json_dict['shapes'][0]['label']
     #   if np.sum(mask) < 100 or np.sum(filet) < 100:
      #      pass
       #     del data[front]
       # else:
        iou_dict[front] = iou
    return data, iou_dict



class Filet_Seg_Dataset(Dataset):
    def __init__(self, mask_dir,img_dir,split, trs_x=[], trs_y=[]):
        self.mask_dict = get_file_pairs(mask_dir,split)
        self.mask_fronts = [key for key in self.mask_dict.keys()]
        self.img_dir = img_dir
        self.prep = PreProcessor([[250,1024-250],[100,1024-100]],resize_dims=(393,618),pad=35,mean=[0,0,0],std=[1,1,1])
        self.split = split
        self.trs_y = trs_y
        self.mask_dir = mask_dir
    def __len__(self):
        return len(self.mask_dict)

    def __getitem__(self, item):
        front = self.mask_fronts[item]
        files = self.mask_dict[front]
        for file in files:
            if file.endswith("mask.jpg"):
                mask_file = os.path.join(self.mask_dir,self.split,file)
                index_to_drop = front.find('_ins')
                img_file = os.path.join(self.img_dir,front[:index_to_drop] + ".jpg")
            elif file.endswith(".json"):
                json_file = os.path.join(self.mask_dir,self.split,file)
            else:
                pass
        mask = np.where(cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)>0,1,0)
        dimh, dimw = mask.shape
        img = cv2.imread(img_file)
#        t1 = time.time()
        img4d = self.prep.preprocess(img,mask)
#        t2 = time.time() - t1
#        print(t2)
        with open(json_file) as fp:
            y = json.load(fp)['shapes'][0]['label']
        for self.tr_y in self.trs_y:
            y = self.tr_y(y)
        return img4d,y



class Filet_Seg_Dataset_Box(Dataset):
    def __init__(self, mask_dir,img_dir,split, trs_x=[], trs_y=[]):
        self.mask_dict = get_file_pairs(mask_dir,split)
        self.mask_fronts = [key for key in self.mask_dict.keys()]
        self.img_dir = img_dir
        self.prep = PreProcessor_Box_Crop([[250,1024-250],[100,1024-100]],resize_dims=(393,618),pad=35,mean=[0.2010, 0.1944, 0.2488, 0.0000],std=[0.3040, 0.2964, 0.3694, 1])
        self.split = split
        self.trs_y = trs_y
        self.mask_dir = mask_dir
    def __len__(self):
        return len(self.mask_dict)

    def __getitem__(self, item):
        front = self.mask_fronts[item]
        files = self.mask_dict[front]
        for file in files:
            if file.endswith("mask.jpg"):
                mask_file = os.path.join(self.mask_dir,self.split,file)
                index_to_drop = front.find('_ins')
                img_file = os.path.join(self.img_dir,front[:index_to_drop] + ".jpg")
            elif file.endswith(".json"):
                json_file = os.path.join(self.mask_dir,self.split,file)
            else:
                pass
        mask = np.where(cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)>0,1,0)
        dimh, dimw = mask.shape
        img = cv2.imread(img_file)
#        t1 = time.time()
        img4d = self.prep.preprocess(img,mask)
#        t2 = time.time() - t1
#        print(t2)
        with open(json_file) as fp:
            y = json.load(fp)['shapes'][0]['label']
        for self.tr_y in self.trs_y:
            y = self.tr_y(y)
        return img4d,y

#dt = Filet_Seg_Dataset("/pers_files/mask_data_raw","/pers_files/Combined_final/Filet/train",'train')
# class Filet_Seg_Dataset(Dataset):
#     '''
#     #transforms bgr to rgb
#     #converts HWC to CHW
#     #converts 0-255 to 0.1
#     '''
#     def __init__(self,file_dict,iou_dict,data_dir,split,trs_x = [], trs_y_bef = [],trs_y_aft = [],mask_only = False ):
#         self.fronts = []
#         self.ious_pre_tr = []
#         self.data_dir = data_dir
#         self.split = split
#         self.mask_only = mask_only
#         self.trs_y_aft = trs_y_aft
#         self.data_dict = copy.deepcopy(file_dict)
#         for front in file_dict.keys():
#             self.fronts.append(front)
#             self.ious_pre_tr.append(iou_dict[front])
#         self.ious = np.array(self.ious_pre_tr)
#         self.ious_pre_tr = np.array(self.ious_pre_tr)
#         self.trs_x = trs_x
#         for tr_y_bef in trs_y_bef:
#             self.ious = tr_y_bef(self.ious)
#         if self.trs_x:
#             self.trs_x = Compose(self.trs_x)
#
#     def __len__(self):
#         return len(self.data_dict)
#
#     def __getitem__(self, item):
#         front = self.fronts[item]
#         files = self.data_dict[front]
#         for file in files:
#             if file.endswith("mask.jpg"):
#                 mask_file = os.path.join(self.data_dir,self.split,file)
#             elif file.endswith(".jpg"):
#                 jpg_file =  os.path.join(self.data_dir,self.split,file)
#             elif file.endswith(".json"):
#                 json_file = os.path.join(self.data_dir,self.split,file)
#             else:
#                 pass
#         pic_load = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
#         pic_load = np.where(pic_load>255/2,1,0)
#         if self.mask_only:
#             dimh,dimw = pic_load.shape
#             pic = np.zeros((dimh, dimw, 1), dtype='uint8')
#             #        with open(os.path.join(data_dir, split, json_file)) as fp:
#             #            json_dict = json.load(fp)
#             iou = self.ious[item]
#             pic = torch.tensor(pic, device='cpu', dtype=torch.float, requires_grad=False).permute(2, 0, 1).contiguous()
#             if not self.trs_x == []:
#                 pic = self.trs_x(pic)
#             for self.tr_y in self.trs_y_aft:
#                 iou = self.tr_y(iou)
#         else:
#             dimh,dimw = pic_load.shape
#             pic = np.zeros((dimh, dimw, 4), dtype='uint8')
#             pic[:, :, 3] = pic_load
#             pic[:, :, :3] = cv2.cvtColor(cv2.imread(jpg_file), cv2.COLOR_BGR2RGB) / 255.0
#             #        with open(os.path.join(data_dir, split, json_file)) as fp:
#             #            json_dict = json.load(fp)
#             iou = np.float(self.ious[item])
#             pic = torch.tensor(pic, device='cpu', dtype=torch.float, requires_grad=False).permute(2, 0, 1)
#             if not self.trs_x == []:
#                 pic = self.trs_x(pic)
#             for self.tr_y in self.trs_y_aft:
#                 iou = self.tr_y(iou)
#                 iou
#         return pic,iou
#
def get_loader(dataset,bs,wts =None):
    if wts is None:
       wts = np.ones(len(dataset))
    sampler = WeightedRandomSampler(weights=wts, num_samples=len(dataset), replacement=True)
    return DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=0, pin_memory=True)

