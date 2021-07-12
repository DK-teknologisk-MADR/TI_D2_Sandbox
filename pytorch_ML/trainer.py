import torch
import os
import time
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader
import numpy as np
class Trainer():
    def __init__(self,dt,net, optimizer = None, scheduler = None,loss_fun = None, max_iter= 200, output_dir ="./trainer_output",eval_period=250, print_period=50,bs=4,dt_val = None,dt_wts = None,fun_val = None , val_nr = None,add_max_iter_to_loaded = False,gpu_id = 0):
        validation_stuff = [dt_val,eval_period,fun_val]
        #validation_variables = {'dt_val' : dt_val, 'eval_period' : eval_period, 'fun_val' : fun_val}
        val_nones = [x is None for x in validation_stuff]
        assert all(val_nones) or not any(val_nones) , "some val variables seems to be none while some are not"
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fun = loss_fun
        self.max_iter = max_iter
        self.output_dir = output_dir
        self.eval_period = eval_period
        self.print_period = print_period
        self.gpu_id = gpu_id
        self.bs = bs
        self.itr = 0
        self.dt = dt
        self.dt_val = dt_val
        self.add_max_iter_to_loaded = add_max_iter_to_loaded
        self.fun_val = fun_val
        self.val_nr = val_nr
        self.dt_wts = dt_wts
        self.best_val_loss = float('inf')
        self.val_loss_cur= float('inf')

        self.net.to('cuda:'+str(self.gpu_id))


    def choose_appropriate_lr_scheduler(self):
        pass

    def get_loader(self,dt,bs,wts = None):
        if wts is None:
            wts = np.ones(len(dt))
        sampler = WeightedRandomSampler(weights=wts, num_samples=len(dt), replacement=True)
        return DataLoader(dt, batch_size=bs, sampler=sampler, pin_memory=True)

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
#        print(torch.cuda.memory_summary(device=self.gpu_id))
        print("TRAINING TO ", self.max_iter)
        self.before_train()
        val_loss = float('-inf')
        done = False
        time_last = time.time()
        while not done:
            if done:
                print("breaking")
                break
            train_dataloader = iter(self.get_loader(self.dt, self.bs,wts=self.dt_wts))
            for batch, targets in train_dataloader:
                time_pt = time.time()
                batch = batch.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                with torch.set_grad_enabled(True):
                    out = self.net(batch).flatten()
                    targets = targets.float()
                    loss = self.loss_fun(out, targets)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if self.itr % self.eval_period == 0 and self.itr > 0 :
                    val_loss = self.validate(self.val_nr)
                    self.net.train()
                    self.scheduler.step(val_loss)
                    self.val_loss_cur = val_loss.numpy()
                    if self.best_val_loss>self.val_loss_cur:
                        self.best_val_loss = self.val_loss_cur

                # log
                if self.itr % self.print_period == 0:
                    time_pt = time.time()
                    print_str = f"time  / iter {(time_pt - time_last) / self.print_period}, iter is {self.itr}, lr is {self.optimizer.param_groups[0]['lr']}, memory allocated is {torch.cuda.memory_allocated(self.gpu_id)}"
                    time_last = time_pt
                    print(print_str)



                done = (self.itr >= self.max_iter)
                self.after_step()
                self.itr = self.itr + 1
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

    def validate(self,val_nr=None, bs = 4,fun = None):
        if fun is None:
            fun = self.fun_val
        self.net.eval()
        if self.dt_val is None:
            raise AttributeError("ERROR, did not supply dt_val but calling validation")
        if val_nr is None:
            val_nr = len(self.dt_val)
        print("validating with ", val_nr, " observations")
        val_loader = self.get_loader(self.dt_val, bs)
        total_loss = torch.tensor(0.0, device=self.gpu_id)
        instances_nr = torch.tensor(0, device='cpu')
        with torch.no_grad():
            for batch, targets in val_loader:
                instances_nr += batch.shape[0]
                batch = batch.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                out = self.net(batch).flatten()
                targets = targets.float()
                total_loss += fun(out, targets)
                if instances_nr + 1 >= val_nr:
                    break
        result = total_loss.to('cpu') / instances_nr
        print("avg val value is " , result)
        return result




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

