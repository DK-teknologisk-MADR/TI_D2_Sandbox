from pytorch_ML.hyperopt import Hyperopt
import numpy as np
from numpy import random
class Mask_Hyperopt(Hyperopt):
    def __init__(self,base_lr,**kwargs):
        super().__init__(**kwargs)
        self.base_lr = base_lr
    def suggest_hyper_dict(self):
        '''
        Overloaded from base
        '''
        generated_values = {
            "optimizer" : { "lr": self.base_lr * random.uniform(1,1000), "momentum": random.uniform(0.1, 0.6)},
            "scheduler" : {'gamma' : None},
            "loss" : {},
            "net" : {'two_layer_head' : random.random()>0.5},
        }
        lr_half_time = random.uniform(2000, 10000)
        generated_values['scheduler']["gamma"] = np.exp(-np.log(2)/lr_half_time)
        return generated_values
