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
        resize_choices = [(224,224),(448,224),(448,448),(224,448)]
        resize_tup = resize_choices[np.random.randint(0,len(resize_choices))]
        pad_choices = [5*i+20 for i in range(6)]
        pad = pad_choices[np.random.randint(0,len(resize_choices))]
        generated_values = {
            "optimizer" : { "lr": self.base_lr * random.uniform(1,100), "momentum": random.uniform(0.1, 0.6)},

            "scheduler" : {'gamma' : None},
            "loss" : {},
            "net" : {'two_layer_head' : random.random()>0.5},
            'resize' : resize_tup,
            'pad' : pad,
        }
        lr_half_time = random.uniform(1000, 7500)
        generated_values['scheduler']["gamma"] = np.exp(-np.log(2)/lr_half_time)
        return generated_values


    def construct_trainer_objs(self,hyper):
        '''
        builds model, optimizer,scheduler and other objects needed for training. Must be deterministic given hyper_dict
        '''
        hyper_objs = super().construct_trainer_objs(hyper)
        dt = self.dt
        prep = PreProcessor_Crop_n_Resize_Box(resize_dims=hyper['resize'],pad=hyper['pad'],mean=[0.2010, 0.1944, 0.2488, 0.0000],std=[0.224, 0.224, 0.224, 1])
        dt.dataset.set_preprocessor(prep)
        hyper_objs['dt'] = dt
        return hyper_objs