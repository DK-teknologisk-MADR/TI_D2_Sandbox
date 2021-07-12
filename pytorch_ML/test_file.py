import shutil

import pytorch_ML.networks
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import pytorch_ML.trainer as trainers
from torch.utils.data import Dataset
import numpy as np
from pytorch_ML.networks import Backbone_And_Fc_Head
from torchvision.transforms import Compose
import os
import shutil
import cv2
import json
trs_x = [ transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
batch_size = 4

def quantitize_cats(ys):
    return np.where(ys == "filet",1,0)

class TI_Dataset(Dataset):
    '''
    #transforms bgr to rgb
    #converts HWC to CHW
    #converts 0-255 to 0.1
    '''
    def __init__(self,data_dir,trs_x = [], trs_y_bef = [],trs_y_aft = [],pre_load_ys = True ):
        self.fronts = []
        self.ys_pre_tr = []
        self.pre_load_ys = pre_load_ys
        self.data_dir = data_dir
        self.trs_y_aft = trs_y_aft
        self.data_dict = self.get_file_pairs()
        for front,files in self.data_dict.items():
            with open(os.path.join(data_dir,front + ".json")) as fp:
                json_dict = json.load(fp)
                self.fronts.append(front)
                y = json_dict['label']
            self.ys_pre_tr.append(y)
        self.ys = np.array(self.ys_pre_tr)
        self.ys_pre_tr = np.array(self.ys_pre_tr)
        self.trs_x = trs_x
        for tr_y_bef in trs_y_bef:
            self.ys = tr_y_bef(self.ys)
        if self.trs_x:
            self.trs_x = Compose(self.trs_x)

    def get_file_pairs(self):
        '''
        input: data directory, and split ( train val test).
        output: dict of pairs of files one image, and one json file.
        '''
        data_dict = {}
        file_to_pair = os.listdir(self.data_dir)
        while (file_to_pair):
            name = file_to_pair.pop()
            front = name.split(".")[0]
            if front in data_dict.keys():
                data_dict[front].append(name)
            else:
                data_dict[front] = [name]
        # drop those where there is not both json and jpg:
        to_drop = []
        for key, data in data_dict.items():
            if not len(data) >= 2:
                to_drop.append(key)
        for key in to_drop:
            data_dict.pop(key)
            print("dropping data", key, " due to missing data")
        return data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        front = self.fronts[item]
        files = self.data_dict[front]
        for file in files:
            if file.endswith(".jpg"):
                jpg_file =  os.path.join(self.data_dir,file)
            elif file.endswith(".json"):
                json_file = os.path.join(self.data_dir,file)
            else:
                print("found weird file",file)
        pic = cv2.imread(jpg_file)
        if self.pre_load_ys:
            y = self.ys[item]
        else:
            with open(os.path.join(json_file)) as fp:
                json_dict = json.load(fp)
                self.fronts.append(front)
                y = json_dict['label']
        pic = torch.tensor(pic, device='cpu', dtype=torch.float, requires_grad=False).permute(2, 0, 1)
        if not self.trs_x == []:
            pic = self.trs_x(pic)
        for self.tr_y in self.trs_y_aft:
            y = self.tr_y(y)
        return pic,y
trs_y_bef = [ quantitize_cats]

net = Backbone_And_Fc_Head(device = 'cuda:0')
x = TI_Dataset(data_dir="/pyscripts/pytorch_ML/test_data/test_filet",trs_x = trs_x,trs_y_bef=trs_y_bef,pre_load_ys=True)
optimizer = optim.SGD(net.parameters(),lr=0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[150,300],gamma=0.5)
loss_fun = torch.nn.BCEWithLogitsLoss()
trainer  = trainers.Trainer_Save_Best(dt = x,net = net,optimizer = optimizer,scheduler= scheduler,loss_fun = loss_fun,max_iter = 200)

trainer.train()












#ls = list(os.walk("./pytorch_ML/test_data/unsorted"))
#idx = 0
#
# for dir,folders,files in ls:
#     for file in files:
#         if file.endswith(".jpg"):
#             record = {}
#             record["file_name"] = os.path.join(os.path.join(dir,file))
#             img = cv2.imread(record["file_name"])
#             record["image_id"] = idx
#             record["height"] = 1024
#             record["width"] = 1024
#             if os.path.basename(dir) == "Filet":
#                 record["label"] = "filet"
#             else:
#                 record["label"] = "spoleben"
#                 img = img[0:1024, 500:(500 + 1024)]
#                 print(img.shape)
#             front = file[:-4]
#             with open(os.path.join(dir,front + ".json"),"w+") as fp:
#                json.dump(record,fp)
#
#             shutil.copy(os.path.join(dir,front + ".json"), f"./pytorch_ML/test_data/test_" + record['label']+"/" + front + ".json")
#             cv2.imwrite(f"./pytorch_ML/test_data/test_" + record['label']+"/" + front + ".jpg", img)
#
#             idx += 1
