import os
import numpy as np
import torch
import torch.nn as nn
import time
from numpy.random import randint, uniform
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage import io, transform
from torch.utils.data import DataLoader
from torchvision import datasets
from pruners import SHA
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
import json
import os
import io
from filet_train.mask_discriminator.mask_data_loader import get_file_pairs,rm_dead_data_and_get_ious , Filet_Seg_Dataset, get_loader
from torchvision.models import resnet101, wide_resnet50_2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#model_resnet = resnet101(True).to(compute_device)
compute_device = "cuda:1"

class IOU_Discriminator(nn.Module):
    def __init__(self):
        super(IOU_Discriminator, self).__init__()
        self. model_wide_res = wide_resnet50_2(True)
        weight = self.model_wide_res.conv1.weight.clone()
        self.model_wide_res.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model_wide_res.to(compute_device)
        self.dropout_feat = nn.Dropout(0.25)
        self.linhead1 = nn.Linear(1000,1)
        with torch.no_grad():
            self.model_wide_res.conv1.weight[:, :3] = weight
            self.model_wide_res.conv1.weight[:, :3]
            self.model_wide_res.conv1.weight[:, 3] = self.model_wide_res.conv1.weight[:, 0]

    def forward(self,x):
        x = self.model_wide_res.forward(x)
        x = self.dropout_feat(x)
        x = self.linhead1(x)
        x = self.linout(x)
        return x

data_dir = '/pers_files/mask_data'
train_split = ""
model_path = 'pers_files/mask_data/models'
val_split = ""
wide_resnet50_2()
file_pairs = get_file_pairs(data_dir,train_split)
data , iou_dict = rm_dead_data_and_get_ious(data_dir,train_split,file_pairs)
dt = Filet_Seg_Dataset(data,iou_dict,data_dir,train_split,[Normalize( mean=[0.485, 0.456, 0.406,0.425], std=[0.229, 0.224, 0.225,0.226])])
dt_val = Filet_Seg_Dataset(data,iou_dict,data_dir,val_split,[Normalize( mean=[0.485, 0.456, 0.406,0.425], std=[0.229, 0.224, 0.225,0.226])])

torch.cuda.empty_cache()
torch.cuda.memory_allocated()

sha = SHA( 300,3,3)




class StopByProgressHook():
    '''
    Classic early stop by tracking progress. Assumes that score can be obtained as (score,key) value from key "storage_key" from trainer storage.
    '''
    def __init__(self,patience,delta_improvement):
        self.should_stop = False
        self.score_best = float('-inf')
        self.score_milestone = float('-inf')
        self.info_best = 0
        self.iter_best = 0
        self.iter_milestone = 0
        self.patience = patience
        self.delta_improvement = delta_improvement
        self.remaining_patience = float('inf')

    def report_score(self, score_cur,iter, info=None):
        is_best = self.score_best < score_cur
        if self.score_best < score_cur:
            self.score_best, self.iter_best, self.info_best = score_cur, iter, info
           # self.save()
            if self.score_milestone < score_cur - self.delta_improvement:
                self.iter_milestone, self.score_milestone = iter, score_cur
            print(self.__str__())
      #      self.trainer.storage.put_scalar(f'best_{self.score_storage_key}', self.score_best, False)
      #      print("got from storage:,",self.trainer.storage.latest()[f'best_{self.score_storage_key}'][0])
        self.remaining_patience = (self.patience- (iter- self.iter_milestone))
        if self.remaining_patience < 0:
            self.should_stop = True
        else:
            self.should_stop = False
        return is_best , self.should_stop




def before_train():
    pass
def after_train():
    pass
def before_step():
    pass
def after_step():
    pass

def save(model, optimizer,scheduler,itr,filename,to_save):
    if filename is None:
        os.makedirs(os.path.join(os.getcwd(), "models" ), exist_ok=True)
        os.path.join(os.getcwd(),"models","checkpoint.pth")
    state = {'itr': itr, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler' : scheduler.state_dict(),
             }
    state.update(to_save)
    torch.save(state, filename)

def load(model, optimizer,scheduler,filename='checkpoint.pth'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt_dict = torch.load(filename)
        model.load_state_dict(ckpt_dict['state_dict'])
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        scheduler.load_state_dict(ckpt_dict['scheduler'])
        print(f"=> loaded checkpoint '{filename}' (itr {ckpt_dict['itr']})")
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    return ckpt_dict


loss_fun = nn.BCEWithLogitsLoss()
def train(net,optimizer,scheduler,max_iter,output_dir,itr = 0,eval_period = 250,print_period = 50,bs = 4):
    loss_fun = nn.BCEWithLogitsLoss()
    before_train()
    done = False
    time_last = time.time()
    while not done:
        train_dataloader = iter(get_loader(dt,bs,0))
        for batch,targets in train_dataloader:
            time_pt = time.time()
            batch = batch.to(compute_device)
            targets = targets.to(compute_device)
            with torch.set_grad_enabled(True):
                out=net(batch).flatten()
                targets = targets.float()
                loss = loss_fun(out,targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if itr % eval_period == 0:
                val_loss = validate(net,dt_val,4)
                scheduler.step(val_loss)

            after_step()

            #log
            if itr % print_period == 0:
                time_pt = time.time()
                print_str = f"time  / iter {(time_pt-time_last)/print_period}, iter is {itr}, lr is {optimizer.param_groups[0]['lr']}"
                time_last = time_pt
                print(print_str)
                print(torch.cuda.memory_summary(1))

            itr = itr + 1

            done = (itr >= max_iter)
            if done:
                break
    save(net,optimizer,scheduler,itr,output_dir,val_loss)
    after_train()
    return val_loss

def validate(net,dt_val,val_nr = None):
    loss_fun_val = nn.BCEWithLogitsLoss(reduction='sum')

    if val_nr is None:
        val_nr = len(dt_val)
    val_loader = get_loader(dt_val,bs,0)
    total_loss = torch.tensor(0.0,device=compute_device)
    instances_nr = torch.tensor(0, device='cpu')
    with torch.no_grad():
        for batch,targets in val_loader:
            instances_nr += batch.shape[0]
            batch = batch.to(compute_device)
            targets = targets.to(compute_device)
            out = net(batch).flatten()
            targets = targets.float()
            total_loss += loss_fun_val(out, targets)
            if instances_nr + 1>=val_nr:
                break
    return torch.mean(total_loss)


def prepare_train(filename,net,optimizer,scheduler,to_load = None):
    ckpt = load(net, optimizer=optimizer, scheduler=scheduler, filename=filename) #loads inplace
    itr = ckpt['itr']
    result = {}
    for key in to_load:
        print(key)
        result[key] = ckpt[key]
    return result
def resume_or_start_train(filename = None,max_iter=1,hyper = {}, bs=4):
        net = IOU_Discriminator().to(compute_device)
        optimizer = optim.SGD(net.parameters(), lr=hyper['lr'], momentum=hyper['momentum'], nesterov=True)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=hyper['gamma'])
        if filename is not None:
            try:
                dict = prepare_train(filename,net,optimizer,scheduler,["itr"])
                itr = dict['itr']
            except:
                itr = 0
                print( "did not see any chk file in dir", filename)
        else:
            itr = 0
        va_Loss = train(net,optimizer,scheduler,max_iter,itr, bs=bs)
        return va_Loss
    #to_load : lst of strings giving keys to extract from load

def hyperopt(base_path):
    bs = 4
    sha=SHA(100,factor=3,topK=9)
    #create paths
    hyper_vals = []
    paths = []
    for i in range(    sha.participants):
        path = os.path.join(base_path,f"model{i}")
        os.makedirs(path)
        paths.append(path)
        hyper = {
        "lr": uniform(0.001, 0.1),
        "momentum": uniform(0.1, 0.6),
        "gamma": uniform(0.1, 0.7),
        }
        hyper_vals.append(hyper)
    trial_id_cur = 0
    pruned = []
    done = False
    while not done:
        val_loss = resume_or_start_train(paths[trial_id_cur],sha.get_cur_res(),hyper_vals[trial_id_cur],bs = 4)
        trial_id_cur, pruned, done = sha.report_and_get_next_trial(-val_loss)

#net.load(torch.load("apath/to/load"))
hyperopt("/pers_files/mask_data/models",400)
