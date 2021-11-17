from pytorch_ML.networks import Classifier, Classifier_Effnet
from efficientnet_pytorch import EfficientNet
from spoleben_train.Mask_discriminator.dataset import Spoleben_Mask_Dataset,Mask_Dataset_Train
import torch
import pytorch_ML.validators
from pytorch_ML.trainer import Trainer
import os.path as path
from pytorch_ML.validators import mcc_score,mcc_with_th
import os
from cv2_utils.cv2_utils import *
from torch.utils.data.dataset import Subset
import torch.nn as nn
from detectron2_ML.data_utils import get_file_pairs
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_ML.validators import f1_score,f1_score_neg,mcc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau , ExponentialLR
from torch.optim import SGD,Adam
import albumentations as A

data_dir = '/pers_files/spoleben/spoleben_09_2021/spoleben_masks_expert/' #INSERT HERE
output_dir = f"{data_dir}output/outputweekend"
os.makedirs(output_dir)
splits = ['train','val']
splits = {split : split for split in splits}
mask_nr = 4
size = (450,450)

augs = A.Compose(
    [A.ColorJitter(always_apply=True,hue=0.025),
     A.HorizontalFlip(p=0.5)]
)
dt = Mask_Dataset_Train(data_dir=path.join(data_dir,splits['train']),mask_nr=mask_nr,augs=augs,size=size)
dt_val = Spoleben_Mask_Dataset(data_dir=path.join(data_dir,splits['val']),mask_nr=mask_nr,size=size)
for item,data in dt.data_path_dict.items():
    if len(data)<6:
        print(item)
#checkout_imgs(tensor_pic_to_imshow_np(imgs[:3])[:,:,::-1])
#dt = Mask_Dataset_Train(data_dir='/pers_files/spoleben/spoleben_09_2021/spoleben_masks_expert/',mask_nr = 4,size=(180,180))
x = DataLoader(dt,batch_size=2)
for img,y in x:
    print(img,y)

bb = Classifier_Effnet(device = 'cuda',model_name = "efficientnet-b2",in_channels=mask_nr + dt.with_depth + 3,num_classes =mask_nr).to('cuda')
optimizer = Adam(params = bb.parameters(),lr=0.0001)
scheduler = ExponentialLR(optimizer=optimizer,gamma=1)
loss_fun = torch.nn.BCEWithLogitsLoss()
trainer = Trainer(dt=dt,net=bb, optimizer = optimizer, scheduler = scheduler,loss_fun = loss_fun, max_iter= 100000, output_dir =output_dir,eval_period=100, print_period=50,bs=10,dt_val = dt_val,dt_wts = None,fun_val =mcc_with_th , val_nr = None,add_max_iter_to_loaded = False,gpu_id = 0)
trainer.train()
