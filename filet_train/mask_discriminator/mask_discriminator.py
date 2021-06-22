import os
import numpy as np
import torch
import torch.nn as nn
import time
from numpy.random import randint, uniform
from pruners import SHA
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
import tkinter
import cv2
import matplotlib.pyplot as plt
import json
import pandas as pd
import os
import io
from filet_train.mask_discriminator.mask_data_loader import get_file_pairs,rm_dead_data_and_get_ious , Filet_Seg_Dataset, get_loader
from torchvision.models import resnet101, wide_resnet50_2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#model_resnet = resnet101(True).to(compute_device)
compute_device = "cuda:1"

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





#net.load(torch.load("apath/to/load"))

class Trainer():
    def __init__(self,dt,net, optimizer, scheduler,loss_fun, max_iter, output_dir,eval_period=250, print_period=50,bs=4,dt_val = None,dt_wts = None,loss_fun_val = None , val_nr = None,add_max_iter_to_loaded = False):
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fun = loss_fun
        self.max_iter = max_iter
        self.output_dir = output_dir
        self.eval_period = eval_period
        self.print_period = print_period
        self.bs = bs
        self.itr = 0
        self.dt = dt
        self.dt_val = dt_val
        self.add_max_iter_to_loaded = add_max_iter_to_loaded
        self.loss_fun_val = loss_fun_val
        self.val_nr = val_nr
        self.dt_wts = dt_wts
        self.best_val_loss = float('inf')
        self.val_loss_cur= float('inf')
    def save_model(self,to_save,file_name = "checkpoint.pth"):
        state = {
             'itr': self.itr,
                'state_dict': self.net.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        state.update(to_save)
        print("saving to ",  os.path.join(self.output_dir,file_name), "iter is",self.itr)
        print(to_save)
        torch.save(state, os.path.join(self.output_dir,file_name))

        #   for key,value in state.items():
        #       torch.save(value,"/pers_files/test_stuff/" + key)

    def load(self, filepath):
        # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
        if os.path.isfile(filepath):
            print("=> loading checkpoint '{}'".format(filepath))
            ckpt_dict = torch.load(filepath)
            self.net.load_state_dict(ckpt_dict['state_dict'])
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            self.scheduler.load_state_dict(ckpt_dict['scheduler'])
            self.itr = ckpt_dict['itr']
            if self.add_max_iter_to_loaded:
                self.max_iter += self.itr
            self.best_val_score = ckpt_dict['val_loss']
            print(f"=> loaded checkpoint '{filepath}' (itr {self.itr}) with val_loss{ckpt_dict['val_loss']}")
        else:
            print("=> no checkpoint found at '{}'".format(filepath))

#        return ckpt_dict

    def train(self):
        print("TRAINING TO ", self.max_iter)
        self.before_train()
        val_loss = float('-inf')
        done = False
        time_last = time.time()
        while not done:
            if done:
                print("breaking")
                break
            train_dataloader = iter(get_loader(self.dt, 4,wts=self.dt_wts))
            for batch, targets in train_dataloader:
                time_pt = time.time()
                batch = batch.to(compute_device)
                targets = targets.to(compute_device)
                with torch.set_grad_enabled(True):
                    out = self.net(batch).flatten()
                    targets = targets.float()
                    loss = self.loss_fun(out, targets)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.itr % self.eval_period == 0:
                    val_loss = self.validate(self.val_nr)
                    self.net.train()
                    self.scheduler.step(val_loss)
                    self.val_loss_cur = val_loss.numpy()
                    if self.best_val_loss>self.val_loss_cur:
                        self.best_val_loss = self.val_loss_cur

                # log
                if self.itr % self.print_period == 0:
                    time_pt = time.time()
                    print_str = f"time  / iter {(time_pt - time_last) / self.print_period}, iter is {self.itr}, lr is {self.optimizer.param_groups[0]['lr']}"
                    time_last = time_pt
                    print(print_str)
                    print(torch.cuda.memory_summary(1,True))


                done = (self.itr >= self.max_iter)

                self.itr = self.itr + 1
                self.after_step()
                if done:
                    print("reached", self.max_iter)
                    break
        self.after_train()

        return self.best_val_loss

    def before_train(self):
        pass

    def after_train(self):
        val_loss = self.validate(self.val_nr)
        self.net.train()
        self.val_loss_cur = val_loss.numpy()
        if self.best_val_loss > self.val_loss_cur:
            self.best_val_loss = self.val_loss_cur
            self.save_model(file_name=f"best_model.pth", to_save={'val_loss': self.best_val_loss})
        self.save_model(file_name=f"checkpoint.pth",to_save={'val_loss' : self.best_val_loss})

    def before_step(self):
        pass

    def after_step(self):
        pass

    def validate(self,val_nr=None, bs = 4):
        self.net.eval()
        if self.dt_val is None:
            raise AttributeError("ERROR, did not supply dt_val but calling validation")
        if val_nr is None:
            val_nr = len(self.dt_val)
        val_loader = get_loader(self.dt_val, bs)
        total_loss = torch.tensor(0.0, device=compute_device)
        instances_nr = torch.tensor(0, device='cpu')
        with torch.no_grad():
            for batch, targets in val_loader:
                instances_nr += batch.shape[0]
                batch = batch.to(compute_device)
                targets = targets.to(compute_device)
                out = self.net(batch).flatten()
                targets = targets.float()
                total_loss += self.loss_fun_val(out, targets)
                if instances_nr + 1 >= val_nr:
                    break
        return total_loss.to('cpu')/instances_nr


class Trainer_Save_Best(Trainer):
    def __init__(self,**kwargs_to_trainer):
        super().__init__(**kwargs_to_trainer)
        self.previous_best_loss = float('inf')

    def after_step(self):
        if self.itr % self.eval_period == 0:
            print("checking if best<pervious",self.best_val_loss<self.previous_best_loss)
            if self.best_val_loss<self.previous_best_loss:
                self.save_model(file_name="best_model.pth",to_save={'val_loss' : self.best_val_loss})
            self.previous_best_loss = self.best_val_loss


class Hyperopt():
    def __init__(self,base_path, max_iter, dt, iter_chunk_size ,model_cls,optimizer_cls, scheduler_cls,loss_cls,output_dir,bs = 4,base_params = {},dt_val = None,eval_period = 250,dt_wts = None, val_nr = None):
        self.base_path = base_path
        self.max_iter = max_iter
        self.bs = bs
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls
        self.loss_cls = loss_cls
        self.dt = dt
        self.dt_wts = dt_wts
        self.eval_period = eval_period
        self.dt_val = dt_val
        self.base_params = base_params
        self.iter_chunk_size = iter_chunk_size
        self.output_dir = output_dir
        self.model_cls = model_cls
        self.pruner = SHA(self.max_iter / self.iter_chunk_size, factor=3, topK=3)
        self.val_nr = val_nr
        self.result_df = None
    def hyperopt(self):
        bs = 4
        # create paths
        hyper_vals = []
        paths = []
        for i in range(self.pruner.participants):
            path = os.path.join(self.base_path, f"model{i}")
            os.makedirs(path, exist_ok=True)
            paths.append(path)
            hyper = self.suggest_hyper_dict()
            hyper_vals.append(hyper)
        self.result_df = pd.DataFrame(hyper_vals)
        self.result_df['val_score'] = np.nan
#        self.result_df['pruned'] = False TODO::Implement pruned into df
        trial_id_cur = 0
        pruned = []
        done = False
        while not done:
            val_loss = self.resume_or_initiate_train(model_dir= paths[trial_id_cur],max_iter= self.iter_chunk_size * self.pruner.get_cur_res(), hyper = hyper_vals[trial_id_cur], bs=4)
            trial_id_cur, pruned, done = self.pruner.report_and_get_next_trial(-val_loss)
            self.result_df.loc[trial_id_cur,'val_score'] = -val_loss
            print("trial sprint completed for ID",trial_id_cur,". rung is ",self.pruner.rung_cur)
            print("result of this sprint was",val_loss )
            print(".........results are................")
            self.pruner.print_status()
        self.result_df.to_csv(os.path.join(self.output_dir,"result.csv"))

    def resume_or_initiate_train(self,model_dir=None, max_iter=1, hyper={}, bs=4):
        optimizer_params = self.base_params['optimizer'] if 'optimizer' in self.base_params else {}
        model_params = self.base_params['model'] if 'model' in self.base_params else {}
        scheduler_params = self.base_params['scheduler'] if 'scheduler' in self.base_params else {}
        loss_params = self.base_params['loss'] if 'loss' in self.base_params else {}

        net = self.model_cls(**model_params).to(compute_device)
        optimizer = self.optimizer_cls(net.parameters(), lr=hyper['lr'], momentum=hyper['momentum'],**optimizer_params)
        scheduler = self.scheduler_cls(optimizer,factor=hyper['gamma'],**scheduler_params)
        loss_fun = self.loss_cls(**loss_params)
        loss_fun_val = self.loss_cls(reduction = 'sum')
        print("initializing training with hyper-parameters",hyper)
        trainer = Trainer_Save_Best(dt = self.dt,dt_wts = self.dt_wts,net = net,optimizer=optimizer,max_iter=max_iter,scheduler=scheduler,loss_fun = loss_fun,output_dir = model_dir,eval_period = self.eval_period,print_period=50,bs=4,dt_val=self.dt_val, loss_fun_val=loss_fun_val,val_nr = self.val_nr,add_max_iter_to_loaded=True)
        if model_dir is not None:
            try:
                trainer.load(os.path.join(model_dir,"checkpoint.pth"))
            except Exception as e:
                print(e)
        try:
            va_Loss = trainer.train()
        except ValueError:
            va_Loss = float('inf')
        return va_Loss
    # to_load : lst of strings giving keys to extract from load

    def suggest_hyper_dict(self):
        '''
        overload this function with hyper generation.

        '''
        hyper = {
            "lr": uniform(0.0005, 0.1),
            "momentum": uniform(0.1, 0.6),
            "gamma": uniform(0.1, 0.9),
        }
        return hyper
