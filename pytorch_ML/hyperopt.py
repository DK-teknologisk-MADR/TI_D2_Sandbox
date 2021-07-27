import numpy as np
from numpy.random import uniform
from detectron2_ML.pruners import SHA
import pandas as pd
import os
from pytorch_ML.trainer import Trainer

#matplotlib.use('TkAgg')
#model_resnet = resnet101(True).to(compute_device)
compute_device = "cuda:0"

def set_device(device):
    compute_device = device
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
        self.hyper_vals = []
        paths = []
        for i in range(self.pruner.participants):
            path = os.path.join(self.output_dir, f"model{i}")
            os.makedirs(path, exist_ok=True)
            paths.append(path)
            hyper = self.suggest_hyper_dict()
            self.hyper_vals.append(hyper)
        self.result_df = pd.DataFrame(self.hyper_vals)
        self.result_df['val_score'] = np.nan
        self.result_df['pruned'] = False
        trial_id_cur = 0
        pruned = []
        done = False
        while not done:
            val_loss = self.resume_or_initiate_train(model_dir= paths[trial_id_cur],max_iter= self.iter_chunk_size * self.pruner.get_cur_res(), hyper = hyper_vals[trial_id_cur], bs=4)
            self.result_df.loc[trial_id_cur,'val_score'] = -val_loss
            trial_id_cur, pruned, done = self.pruner.report_and_get_next_trial(-val_loss)
            print("Hyperopt : trial sprint completed for ID",trial_id_cur,". rung is ",self.pruner.rung_cur)
            print("Hyperopt : result of this sprint was",val_loss )
            print("Hyperopt : .........results are................")
            self.result_df.loc[pruned,'pruned'] = True
            self.after_prune()
            self.pruner.print_status()
        self.result_df.to_csv(os.path.join(self.output_dir,"result.csv"))

    def resume_or_initiate_train(self,model_dir=None, max_iter=1, hyper={}, bs=4):
        #TODO: MERGE THIS FROM TI not pushed to GIT yet
        optimizer_params = self.base_params['optimizer'] if 'optimizer' in self.base_params else {}
        model_params = self.base_params['model'] if 'model' in self.base_params else {}
        scheduler_params = self.base_params['scheduler'] if 'scheduler' in self.base_params else {}
        loss_params = self.base_params['loss'] if 'loss' in self.base_params else {}
        net = self.model_cls(**model_params).to(compute_device)
        optimizer = self.optimizer_cls(net.parameters(), lr=hyper['lr'], momentum=hyper['momentum'],**optimizer_params)
        scheduler = self.scheduler_cls(optimizer,factor=hyper['gamma'],**scheduler_params)
        loss_fun = self.loss_cls(**loss_params)
        fun_val = self.loss_cls(reduction = 'sum')
        trainer = Trainer(dt = self.dt,dt_wts = self.dt_wts,net = net,optimizer=optimizer,max_iter=max_iter,scheduler=scheduler,loss_fun = loss_fun,output_dir = model_dir,eval_period = self.eval_period,print_period=50,bs=self.bs,dt_val=self.dt_val, fun_val=fun_val,val_nr = self.val_nr,add_max_iter_to_loaded=True)
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

    def after_prune(self):
        pass

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

