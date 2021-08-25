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
        base_resize = (400,620)
        resize_scales = [0.33,0.5,0.75,1]
        chosen_resize_scale_index = np.random.randint(0,len(resize_scales))
        chosen_scale = resize_scales[chosen_resize_scale_index]
        resize_tup = [int(chosen_scale*base_resize[0]),int(chosen_scale * base_resize[1])]
        is_resize_sq = np.random.randint(0, 2)
        if is_resize_sq:
            resize_tup[0] = ( resize_tup[0] + resize_tup[1] ) // 2
            resize_tup[1] = resize_tup[0]
        pad_choices = [5*i+25 for i in range(6)]
        pad = pad_choices[np.random.randint(0,len(pad_choices))]
        generated_values = {
            "optimizer" : { "lr": self.base_lr * random.uniform(1,100), "momentum": random.uniform(0.1, 0.6)},

            "scheduler" : {'gamma' : None},
            "loss" : {},
            "net" : {'two_layer_head' : random.random()>0.5,},
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