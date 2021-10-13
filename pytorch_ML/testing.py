import torchvision
import os
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_ML.networks import Classifier
import torch.nn as nn
import shutil
import numpy as np
from pytorch_ML.validators import f1_score,f1_score_neg,mcc_score
from detectron2_ML.pruners import SHA
from torch.utils.data import Dataset
from matplotlib.pyplot import imshow
#tr=torchvision.transforms.Resize(228)
#tr2 = torchvision.transforms.ToTensor()
#tr = torchvision.transforms.Compose([tr,tr2])
tr = torchvision.transforms.ToTensor()
target_tr = lambda x : 1 if x== 6 else 0
data_train = torchvision.datasets.CIFAR10(root="/pers_files/test_data",train=True,transform=tr,target_transform=target_tr,download=True)
data_test = torchvision.datasets.CIFAR10(root="/pers_files/test_data",train=False,transform=tr,target_transform=target_tr,download=True)
labels= np.zeros(2)
wts = np.ones(50000)
for i,tup in enumerate(data_train):
    x,y = tup
    if y == 1:
        wts[i] = 9
wts = wts / 10
print(labels)
model_path = "/pers_files/test_model"
from pytorch_ML.hyperopt import Hyperopt
import numpy as np
from numpy import random
from filet.mask_discriminator.transformations import PreProcessor_Crop_n_Resize_Box

class Mask_Hyperopt(Hyperopt):
    def __init__(self,base_lr,**kwargs):
        super().__init__(**kwargs)
        self.base_lr = base_lr
    def suggest_hyper_dict(self):
        '''
        Overloaded from base
        '''
        base_resize = (228,228)
        resize_scales = [0.5,1]
        chosen_resize_scale_index = np.random.randint(0,len(resize_scales))
        chosen_scale = resize_scales[chosen_resize_scale_index]
        resize_tup = [int(chosen_scale*base_resize[0]),int(chosen_scale * base_resize[1])]
        is_resize_sq = np.random.randint(0, 2)
        if is_resize_sq:
            resize_tup[0] = ( resize_tup[0] + resize_tup[1] ) // 2
            resize_tup[1] = resize_tup[0]
        pad_choices = [5*i+10 for i in range(1)]
        pad = pad_choices[np.random.randint(0,len(pad_choices))]
        generated_values = {
            "optimizer" : { "lr": self.base_lr * random.uniform(1,100), "momentum": random.uniform(0.1, 0.6)},

            "scheduler" : {'gamma' : None},
            "loss" : {},
            'resize' : resize_tup,
            'pad' : pad,
        }
        lr_half_time = random.uniform(1000, 7500)
        generated_values['scheduler']["gamma"] = 1#        generated_values['scheduler']["gamma"] = np.exp(-np.log(2)/lr_half_time)
        return generated_values


    def construct_trainer_objs(self,hyper):
        '''
        builds model, optimizer,scheduler and other objects needed for training. Must be deterministic given hyper_dict
        '''
        hyper_objs = super().construct_trainer_objs(hyper)
        dt = self.dt
     #   prep = PreProcessor_Crop_n_Resize_Box(resize_dims=hyper['resize'],pad=hyper['pad'],mean=[0.2010, 0.1944, 0.2488, 0.0000],std=[0.224, 0.224, 0.224, 1])
     #  dt.dataset.set_preprocessor(prep)
        hyper_objs['dt'] = dt

        return hyper_objs

# model_path = "~/Pyscripts/test_folder"
base_params = {
    "optimizer": {
        'nesterov': True,
    },
    "scheduler": {},
    "optimizer_cls": SGD,
    "scheduler_cls": ExponentialLR,
    "loss_cls": nn.BCEWithLogitsLoss,
    "net_cls": Classifier,
    "net": {'device': 'cuda:1'}
}

max_iter, iter_chunk_size = 4*3000000 + 1,3000000
pruner = SHA(max_iter / iter_chunk_size, factor=4, topK=4)
output_dir = os.path.join(model_path, "test_model1")
shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
hyper = Mask_Hyperopt(base_lr=0.0005, base_path=model_path, max_iter=max_iter,dt_wts=wts, iter_chunk_size=iter_chunk_size,
                      dt=data_train, output_dir=output_dir, val_nr=None, bs=4, base_params=base_params, dt_val=data_test,
                      eval_period=200, fun_val=mcc_score, pruner=pruner, gpu_id=0)
hyper.hyperopt()
