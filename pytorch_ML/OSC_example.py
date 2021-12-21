import torchvision
import os
import torch
import scipy
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_ML.networks import Classifier,Classifier_Multi_Effnet
from torch.utils.data import DataLoader
from pytorch_ML.data_utils import get_balanced_class_weights_from_dataset
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pytorch_ML.validators import f1_score,f1_score_neg,mcc_score, F1_Score_Osc
from detectron2_ML.pruners import SHA
from torch.utils.data import Dataset
from matplotlib.pyplot import imshow
from torchvision.transforms import ToTensor
from pytorch_ML.trainer import Trainer
import albumentations as A
from torch.optim import SGD,Adam
from torch.utils.data import Subset
from cv2_utils.cv2_utils import *
from pytorch_ML.trainer import strict_batch_collater
from pytorch_ML.osc_loss import OSC_Loss

nr_of_classes = 10
target_tr = lambda x : x if x < nr_of_classes else nr_of_classes

class my_transform():
    def __init__(self):
        transforms = [A.ColorJitter(hue=0.05),
        A.ShiftScaleRotate(p=0.25,scale_limit=0.05),
        A.Blur(blur_limit=3,p=0.25),
        A.CoarseDropout(p=0.25,max_holes= 4,min_holes=2,max_height=16,min_height=8,max_width=16,min_width=8),
        A.HorizontalFlip(p=0.5),
        A.Resize(300, 300),
        A.RandomCrop(height=280,width=280,p=0.25),
        A.Resize(300, 300),
                      ]
        self.tr = A.Compose(transforms)
        self.tr_ts = ToTensor()

    def __call__(self,img):
        img = np.array(img)
        img = self.tr(image=img)['image']
        img_ts = self.tr_ts(img)
        return img_ts


tr = my_transform()

dataset = torchvision.datasets.Caltech101(root="/pers_files/caltech/",transform=tr,target_transform=target_tr,download=False)
color_pics = []
for i,data in enumerate(dataset):
    img,y = data
    if img.shape[0] == 3:
        color_pics.append(i)
dataset =Subset(dataset,color_pics)

train_indices,val_indices,test_indices = [],[],[]
for i in range(len(dataset)):
    if i % 10 == 0:
        test_indices.append(i)
    elif i % 10 == 1:
        val_indices.append(i)
    else:
        train_indices.append(i)
data_train,data_val,data_test = Subset(dataset,train_indices),Subset(dataset,val_indices),Subset(dataset,test_indices)


probs = np.full(nr_of_classes + 1 ,3/4*(1/nr_of_classes))
probs[-1] = 1/4
weights = get_balanced_class_weights_from_dataset(dataset=data_train,probs = probs)

for i,data in enumerate(data_train):
    if len(data)!= 2:
        print(i,len(data))
for i, data in enumerate(data_val):
    if len(data)!= 2:
        print(i,len(data))

#bb = EfficientNet.from_name(model_name='efficientnet-b3',in_channels=3,num_classes=nr_of_classes)
bb = Classifier_Multi_Effnet(device="cuda:1",pretrained = True, model_name='efficientnet-b3',in_channels=3,num_classes=nr_of_classes)

optimizer = Adam(params = bb.parameters(),lr=0.0001)
scheduler = ExponentialLR(optimizer=optimizer,gamma=1)
loss_fun = OSC_Loss(nr_of_classes=10)
val_fun = F1_Score_Osc(10,th=0.3)
output_dir = "/pers_files/OSC_TEST"
trainer = Trainer(dt=data_train,net=bb, optimizer = optimizer, scheduler = scheduler,loss_fun = loss_fun, max_iter= 1000, output_dir =output_dir,eval_period=50, print_period=50,bs=10,dt_val = data_val,dt_wts = weights,fun_val =val_fun , val_nr = None,add_max_iter_to_loaded = False,gpu_id = 1,unsqueeze_ys=True)
trainer.train()


#xs,ys = batch
#y_hat = bb(xs.to('cuda'))
#loss_fun(targets=ys.to('cuda'),outs=y_hat)
#y_hat.shape
#z = np.zero1s((1,10,10))
#z[:,2:7,2:7] = 1
#.shape