import shutil

import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from numpy.random import uniform
import copy
import pytorch_ML.networks
from detectron2_ML.pruners import SHA
import pandas as pd
import os
import torch.nn as nn
import random
import torch.optim as optim
from pytorch_ML.trainer import Trainer
from pytorch_ML.validators import f1_score
#matplotlib.use('TkAgg')
#model_resnet = resnet101(True).to(compute_device)
compute_device = "cuda:1"
import json
def set_device(device):
    compute_device = device


class Hyperopt():
    def __init__(self,base_path, max_iter, dt, iter_chunk_size ,output_dir,bs = 4,base_params = {},dt_val = None,eval_period = 250,dt_wts = None, val_nr = None,fun_val = None,pruner=None,gpu_id = 0):
        assert fun_val is not None , "please supply a fun val"
        self.base_path = base_path
        self.max_iter = max_iter
        set_device('cuda:' + str(gpu_id))
        self.bs = bs
        self.dt = dt
        self.gpu_id = gpu_id
        self.dt_wts = dt_wts
        self.eval_period = eval_period
        self.dt_val = dt_val
        self.base_params = base_params
        self.iter_chunk_size = iter_chunk_size
        self.output_dir = output_dir
        if pruner is None:
            self.pruner = SHA(self.max_iter / self.iter_chunk_size, factor=3, topK=9)
        else:
            self.pruner = pruner
        self.val_nr = val_nr
        self.result_df = None
        self.fun_val = fun_val


    def suggest_hyper_dict(self):
        '''
        Generates hyper-values. Should return them in nested_dicts.
        All sub dictionaries of output_dictwill be merged with base param sub-dictionaries.
         In case of duplicate, the output of this function wins
        overload this function with hyper generation.
        '''
        generated_values = {
            "optimizer" : { "lr": None, "momentum": uniform(0.1, 0.6)},
            "scheduler" : {'gamma' : None},
            "loss" : {},
            "net" : {'two_layer_head' : random.random()>0.5},
        }
        lr_half_time = uniform(200, 4000)
        generated_values['scheduler']["gamma"] = np.exp(-np.log(2)/lr_half_time)
        lr_magnitude = - random.randint(2,6)
        lr_scale = random.uniform(1,10)
        generated_values['optimizer']["lr"] = 10**(lr_magnitude) * lr_scale
        return generated_values


    def construct_trainer_objs(self,hyper):
        '''
        builds model, optimizer,scheduler and other objects needed for training. Must be deterministic given hyper_dict
        '''
        net = hyper["net_cls"](**hyper['net'])
        optimizer = hyper["optimizer_cls"](net.parameters(),**hyper["optimizer"])
        scheduler = hyper["scheduler_cls"](optimizer,**hyper["scheduler"])
        loss_fun = hyper['loss_cls'](**hyper['loss'])
        fun_val = self.fun_val
        hyper_objs = {
            'optimizer' : optimizer,
            'scheduler' : scheduler,
            'loss_fun' : loss_fun,
            'net' : net,
        }
        return hyper_objs

    def hyper_dict_to_json_parsable_dict(self,hyper):
        new_dict = {}
        stack = list(hyper.items())
        visited = set()
        while stack:
            k, v = stack.pop()
            if isinstance(v, dict):
                if k not in visited:
                    stack.extend(v.items())
            else:
                try:
                    test = json.dumps(v)
                    new_dict[k] = v
                except TypeError:
                    new_dict[k] = str(v)
                except:
                    raise TypeError("it seems like its not possible to call str on object", v)
            visited.add(k)
        return new_dict

    def combine_generated_with_base(self,generated_vals):
        #updates
        hyper = copy.deepcopy(self.base_params)

        for key,vals in generated_vals.items():
            if key in hyper:
                if isinstance(vals,dict):
                    if isinstance(hyper[key],dict):
                        hyper[key].update(vals)
                    else:
                        raise ValueError(f"Seems like {key} is a dictionary for hyper-parameters, but not for base parameters")
                else:
                    if isinstance(hyper[key],dict):
                        raise ValueError(f"Seems like {key} is a dictionary for base-parameters, but not for hyper parameters")
                    hyper[key] = vals
            else:
                hyper[key] = vals
        return hyper


    def hyperopt(self):
        bs = 4
        # create paths
        self.hyper_vals = []
        paths = []
        for i in range(self.pruner.participants):
            path = os.path.join(self.output_dir, f"model{i}")
            os.makedirs(path, exist_ok=True)
            paths.append(path)
            generated_values = self.suggest_hyper_dict()
            hyper = self.combine_generated_with_base(generated_values)
            self.hyper_vals.append(hyper)
            #test
        self.result_df = pd.DataFrame(self.hyper_vals)
        self.result_df['val_score'] = np.nan
        self.result_df['pruned'] = False
        trial_id_cur = 0
        pruned = []
        done = False
        while not done:
            score = self.resume_or_initiate_train(model_dir= paths[trial_id_cur],max_iter= self.iter_chunk_size * self.pruner.get_cur_res(), hyper = self.hyper_vals[trial_id_cur], bs=4)
            self.result_df.loc[trial_id_cur,'val_score'] = score
            print("Hyperopt : trial sprint completed for ID",trial_id_cur)
            trial_id_cur, pruned, done = self.pruner.report_and_get_next_trial(float(score))
            print("rung is now",self.pruner.rung_cur)
            print("Hyperopt : result of this sprint was",score )
            print("Hyperopt : .........results are................")
            self.result_df.loc[pruned,'pruned'] = True
            self.after_prune(pruned)
            self.pruner.print_status()
        self.result_df.to_csv(os.path.join(self.output_dir,"result.csv"))

    def resume_or_initiate_train(self,model_dir=None, max_iter=1, hyper={}, bs=4):
        #TODO: TAKE THIS when merging
        hyper_objs = self.construct_trainer_objs(hyper)
        if not 'dt' in hyper_objs:
            hyper_objs['dt'] = self.dt
        if not 'dt_wts' in hyper_objs:
            hyper_objs['dt_wts'] = self.dt_wts
        print("resuming training of",os.path.basename(model_dir))
        print("hyper parameters are:")
        print(hyper)
        trainer = Trainer(max_iter=max_iter,output_dir = model_dir,eval_period = self.eval_period,print_period=50,bs=self.bs,dt_val=self.dt_val, fun_val=self.fun_val,
                          val_nr = self.val_nr,add_max_iter_to_loaded=True,**hyper_objs,gpu_id=self.gpu_id)
        if model_dir is not None:
            try:
                trainer.load(os.path.join(model_dir,"checkpoint.pth"))
            except Exception as e:
                print(e)
        try:
           score = trainer.train()
        except ValueError:
            score = -float('inf')
        return score
    # to_load : lst of strings giving keys to extract from load

    def after_prune(self,pruned):
        '''
        Overwrite this if yo uwant different behavior
        '''
        for prune_id in pruned:
            shutil.rmtree(os.path.join(self.output_dir,f"model{prune_id}"))

























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



#hyper = Hyperopt(None,max_iter = 250000,iter_chunk_size = 100,dt= None,output_dir=os.path.join(model_path,"classi_net"), bs = 3,base_params= base_params,dt_val = None,eval_period = 180,dt_wts = None)

