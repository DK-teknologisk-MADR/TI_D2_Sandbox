from pytorch_ML.networks import Classifier
from efficientnet_pytorch import EfficientNet
from spoleben_train.Mask_discriminator.dataset import Spoleben_Mask_Dataset,Mask_Dataset_Train
import torch
import pytorch_ML.networks
import pytorch_ML.validators
from pytorch_ML.trainer import Trainer
import os.path as path
import os
from torch.utils.data.dataset import Subset
import torch.nn as nn
from detectron2_ML.data_utils import get_file_pairs
import numpy as np
import torch.optim as optim
from pytorch_ML.validators import f1_score,f1_score_neg,mcc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau , ExponentialLR
from torch.optim import SGD
import albumentations as A


data_dir = "" #INSERT HERE
output_dir = ""
splits = ['train','val']
splits = {split : split for split in splits}
augs = A.Compose(
    [A.ColorJitter(p=0.8),
     A.HorizontalFlip(p=0.5)]
)

bb = EfficientNet.from_pretrained("efficientnet-b2",in_channels=5,num_classes = 1).to('cuda')
net = Classifier(backbone=bb)
dt_train = Mask_Dataset_Train(data_dir=path.join(data_dir,splits['train']),augs = augs)
dt_val = Spoleben_Mask_Dataset(data_dir=path.join(data_dir,splits['val']))
optimizer = SGD(params = net.parameters(),lr=0.005,nesterov=True,momentum=0.5)
scheduler = ExponentialLR(optimizer=optimizer,gamma=0.99999)
loss_fun = torch.nn.BCEWithLogitsLoss()
trainer = Trainer(dt=dt_train,net=net, optimizer = optimizer, scheduler = scheduler,loss_fun = loss_fun, max_iter= 500, output_dir =output_dir,eval_period=250, print_period=50,bs=4,dt_val = dt_val,dt_wts = None,fun_val = None , val_nr = None,add_max_iter_to_loaded = False,gpu_id = 0)


