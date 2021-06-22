import numpy as np
import torch
from torchvision.transforms import ToTensor, Compose, Normalize
import cv2

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from data_utils import get_file_pairs
import json
import os
import shutil
from data_utils import get_file_pairs
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def get_file_pairs(data_dir,split):
    '''
    input: data directory, and split ( train val test).
    output: dict of pairs of files one image, and one json file.
    '''
    data_dict = {}
    file_to_pair = os.listdir(os.path.join(data_dir,split))
    while(file_to_pair):
        name = file_to_pair.pop()
        front = name.split(".")[0]
        if front.endswith("mask"):
            front = front[:-4]
        else:
            front = front
        if front in data_dict.keys():
            data_dict[front].append(name)
        else:
            data_dict[front] = [name]
#drop those where there is not both json and jpg:
    to_drop = []
    for key,data in data_dict.items():
        if not len(data) >=2:
            to_drop.append(key)
    for key in to_drop:
        data_dict.pop(key)
        print("dropping data", key, " due to missing data")
    return data_dict




def rm_dead_data_and_get_ious(data_dir,split,file_pairs = None):
    if file_pairs is None:
        file_pairs = get_file_pairs(data_dir,split)
    data = file_pairs.copy()
    i = 0
    iou_dict = {}
    for front,files in file_pairs.items():
        if len(files)< 3:
            print(front,files)
            i+=1
            del data[front]
        else:
            for file in files:
                if file.endswith("mask.jpg"):
                    mask_file = os.path.join(data_dir,split,file)
                elif file.endswith(".jpg"):
                    jpg_file =  os.path.join(data_dir,split,file)
                elif file.endswith(".json"):
                    json_file = os.path.join(data_dir,split,file)
                else:
                    print("warning, cant recognize file",file)
            mask = cv2.imread(mask_file)
            filet = cv2.imread(jpg_file)
            with open(os.path.join(data_dir,split,json_file)) as fp:
                json_dict = json.load(fp)
                iou = json_dict['shapes'][0]['label']
            if np.sum(mask) < 100 or np.sum(filet) < 100:
                del data[front]
                i += 1
            else:
                iou_dict[front] = iou
    print("removed ", i , "dead files")
    return data, iou_dict
#tup = len(files[0])*.8,len(files[0])*.1,len(files[0])*.1
#ls = [int(x) for x in tup]
#files,ious = rm_dead_data_and_get_ious("/pers_files/mask_data/","")
#splits = np.repeat(["train","val","test"],ls)

#for fr_file_pair,split in zip(files.items(),splits):
#    front,files = fr_file_pair
#    for file in files:
#        shutil.copyfile( os.path.join("/pers_files/mask_data/", "", file ), os.path.join("/pers_files/mask_data/", split, file ) )

class Filet_Seg_Dataset(Dataset):
    '''
    #transforms bgr to rgb
    #converts HWC to CHW
    #converts 0-255 to 0.1
    '''
    def __init__(self,file_dict,iou_dict,data_dir,split,trs_x = [], trs_y_bef = [],trs_y_aft = [] ):
        self.fronts = []
        self.ious = []
        self.data_dir = data_dir
        self.split = split
        self.trs_y_aft = trs_y_aft
        self.data_dict = file_dict
        for front in file_dict.keys():
            self.fronts.append(front)
            self.ious.append(iou_dict[front])
        self.ious = np.array(self.ious)
        self.trs_x = trs_x
        for tr_y_bef in trs_y_bef:
            self.ious = tr_y_bef(self.ious)
        if self.trs_x:
            self.trs_x = Compose(self.trs_x)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        front = self.fronts[item]
        files = self.data_dict[front]
        for file in files:
            if file.endswith("mask.jpg"):
                mask_file = os.path.join(self.data_dir,self.split,file)
            elif file.endswith(".jpg"):
                jpg_file =  os.path.join(self.data_dir,self.split,file)
            elif file.endswith(".json"):
                json_file = os.path.join(self.data_dir,self.split,file)
            else:
                pass
        pic_load = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        pic_load = np.where(pic_load>255/2,1,0)
        dim = pic_load.shape[0]
        pic = np.zeros((dim,dim,4),dtype='uint8')
        pic[:,:,3] = pic_load
        pic[:,:,:3] = cv2.cvtColor(cv2.imread(jpg_file),cv2.COLOR_BGR2RGB) / 255.0
#        with open(os.path.join(data_dir, split, json_file)) as fp:
#            json_dict = json.load(fp)
        iou = self.ious[item]
        pic = torch.tensor(pic,device='cpu',dtype=torch.float,requires_grad=False).permute(2,0,1).contiguous()
        if not self.trs_x == []:
            pic = self.trs_x(pic)
        for self.tr_y in self.trs_y_aft:
            iou = self.tr_y(iou)
        return pic,iou

def get_loader(dataset,bs,wts =None):
    if wts is None:
        wts = np.ones(len(dataset))
    sampler = WeightedRandomSampler(weights=wts, num_samples=len(dataset), replacement=True)
    return DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=0, pin_memory=True)

