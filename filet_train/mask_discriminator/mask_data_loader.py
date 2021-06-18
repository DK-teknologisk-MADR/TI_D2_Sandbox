import numpy as np
import torch
from torchvision.transforms import ToTensor, Compose, Normalize
import cv2

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from data_utils import get_file_pairs
import json
import os

from data_utils import get_file_pairs
import matplotlib
matplotlib.use('TkAgg')
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




def rm_dead_data_and_get_ious(data_dir,split,file_pairs):
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


class Filet_Seg_Dataset(Dataset):
    def __init__(self,file_dict,iou_dict,data_dir,split,trs_x = [], trs_y = [] ):
        self.fronts = []
        self.ious = []
        self.data_dir = data_dir
        self.split = split
        self.data_dict = file_dict
        for front in file_dict.keys():
            self.fronts.append(front)
            self.ious.append(iou_dict[front])
        self.trs_x = trs_x
        self.trs_y = trs_y
        if self.trs_x:
            self.trs_x = Compose(self.trs_x)
        if self.trs_y:
            self.trs_y = Compose(self.trs_y)

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
        pic = torch.tensor(pic,device='cpu',dtype=torch.float).permute(2,0,1).contiguous()
        if not self.trs_x == []:
            pic = self.trs_x(pic)
        if not self.trs_y == []:
            iou = self.trs_y(iou)
        return pic,iou

def get_loader(dataset,bs,num_workers):
    ious = dataset.ious
    st_inds = np.argsort(ious)
    qs = np.array([0.86, 0.935])
    intervals = np.searchsorted(qs, ious)
    nrs = np.bincount(intervals)
    weights_pre = nrs[0] / nrs
    weights_pre = weights_pre * (1 / min(weights_pre))
    weights = weights_pre[intervals]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
    return DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=6, pin_memory=True)

