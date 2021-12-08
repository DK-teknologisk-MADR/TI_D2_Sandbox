import torchvision
import os
import torch
import scipy
import os.path as path
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_ML.networks import Classifier,Classifier_Multi_Effnet,ConvAutoencoder
from torch.utils.data import DataLoader,Dataset
from pytorch_ML.data_utils import get_balanced_class_weights_from_dataset
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_ML.validators import f1_score,f1_score_neg,mcc_score, F1_Score_Osc
from detectron2_ML.pruners import SHA
from torch.utils.data import Dataset
import matplotlib
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
matplotlib.use('tkAgg')
from torchvision.transforms import ToTensor
from pytorch_ML.trainer import Trainer
import albumentations as A
from torch.optim import SGD,Adam
from torch.utils.data import Subset
from cv2_utils.cv2_utils import *

nr_of_classes = 10
target_tr = lambda x : x if x<nr_of_classes else nr_of_classes

class my_transform():
    def __init__(self):
        transforms = [A.ColorJitter(hue=0.05),
        A.ShiftScaleRotate(p=0.25,scale_limit=0.00),
        A.HorizontalFlip(p=0.5),
        A.Resize(240, 240),
        A.RandomCrop(height=230,width=230,p=0.25),
        A.Resize(240, 240),
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
class Dataset_Unsupervised(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        x,y = self.dataset[item]
        return x,x.clone()

    def __len__(self):
        return len(self.dataset)

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
data_train,data_val,data_test = Dataset_Unsupervised(data_train),Dataset_Unsupervised(data_val),Dataset_Unsupervised(data_test)




#targets = np.array([y for img,y in data_train])
#indices, counts = np.unique(targets,return_counts=True)
#checkout_imgs( tensor_pic_to_imshow_np(data_train[550][0]),'rgb')

bb = ConvAutoencoder(3,16)


class MSE_score():
    def __init__(self):
        self.loss = nn.MSELoss()
    def __call__(self, x,y):
        return -self.loss(x,y)**0.5



loss_fun = nn.MSELoss()
val_fun = MSE_score()
optimizer = Adam(params = bb.parameters(),lr=0.0001)
scheduler = ExponentialLR(optimizer=optimizer,gamma=1)
output_dir = "/pers_files/encoder_TEST"
trainer = Trainer(dt=data_train,net=bb, optimizer = optimizer, scheduler = scheduler,loss_fun = loss_fun, max_iter= 100000, output_dir =output_dir,eval_period=100, print_period=50,bs=16,dt_val = data_val,dt_wts = None,fun_val =val_fun , val_nr = None,add_max_iter_to_loaded = False,gpu_id = 1,unsqueeze_ys=False)
trainer.train()

bb = ConvAutoencoder(3,16)


class MSE_score():
    def __init__(self):
        self.loss = nn.MSELoss()
    def __call__(self, x,y):
        return -self.loss(x,y)**0.5



loss_fun = nn.MSELoss()
val_fun = MSE_score()
optimizer = Adam(params = bb.parameters(),lr=0.0001)
scheduler = ExponentialLR(optimizer=optimizer,gamma=1)
output_dir = "/pers_files/encoder_TEST"
trainer = Trainer(dt=data_train,net=bb, optimizer = optimizer, scheduler = scheduler,loss_fun = loss_fun, max_iter= 100000, output_dir =output_dir,eval_period=100, print_period=50,bs=16,dt_val = data_val,dt_wts = None,fun_val =val_fun , val_nr = None,add_max_iter_to_loaded = False,gpu_id = 1,unsqueeze_ys=False)
trainer.train()
from pytorch_ML.trainer import strict_batch_collater
#dl = DataLoader(dataset=data_train,batch_size=1,sampler=WeightedRandomSampler(weights=weights,num_samples=len(data_train)),collate_fn=strict_batch_collater)
#dl = iter(dl)
#batch = next(dl)
#loss_fun(outs=bb(batch[0].to('cuda:1')),targets=batch[1].to('cuda:1'))
bb.eval()
with torch.no_grad():
     out = bb(img)
img1 = tensor_pic_to_imshow_np(img.squeeze(0))
img2 = tensor_pic_to_imshow_np(out.squeeze(0))
x = Image.fromarray(img1)
x.show()
x = Image.fromarray(img2)
x.show()
plt.imshow(img1)

