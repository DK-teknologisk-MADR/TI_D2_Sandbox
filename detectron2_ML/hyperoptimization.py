import os
from detectron2.data import build_detection_test_loader
from detectron2_ML.trainers import TI_Trainer, Hyper_Trainer
from detectron2.evaluation import inference_on_dataset
import detectron2_ML.hooks as hooks
from detectron2_ML.hooks import StopAtIterHook
from detectron2_ML.pruners import SHA
import pandas as pd
# install dependencies:
import datetime
#assert torch.cuda.is_available(), "torch cant find cuda. Is there GPU on the machine?"
from detectron2.utils.logger import setup_logger

setup_logger()


class D2_hyperopt_Base():
    '''
    does hyper-optimization for detectron2 models over cfg parameters.

    input:
      task: Possible choices are at time of writing "bbox", "segm", "keypoints".
      evaluator: Use COCOEvaluator if in doubt
      https://detectron2.readthedocs.io/en/latest/modules/evaluation.html#detectron2.evaluation.COCOEvaluator
      step_chunk_size: nr of iters corresponding to 1 ressource granted by pruner
      max_iter : maximum TOTAL number of iters across all tried models. WARNING: large max_iter requires large amount of free space.
      pruner_cls : class(not object) of a pruner. see pruner class
    output: Pandas df of all trials, whether they have been pruned, and their last reported score. This df is also written as csv to output_dir
    '''
    def __init__(self, model_name,cfg_base,data_val_name, task,evaluator,output_dir, step_chunk_size=30,
                 max_iter = 90,
                 trainer_cls = Hyper_Trainer,
                 pruner_cls = SHA,
                 pr_params = {},
                 trainer_params = {},
                 score_name='AP'):
        self.step_chunk_size = step_chunk_size
        self.model_name=model_name
        self.task = task
        self.cfg_base = cfg_base
        self.score_name = score_name
        self.trainer_cls = trainer_cls
        self.suggested_cfgs = []
        self.data_val_name = data_val_name
        self.suggested_params = []
        self.output_dir=output_dir
        self.evaluator = evaluator
        self.pruner = pruner_cls(max_iter // self.step_chunk_size, **pr_params)
        self.date = datetime.date.today()
        self.time = datetime.datetime.now()
        self.time_info_str = "-".join(
            [str(x) for x in [self.date.year, self.date.month, self.date.day, self.time.hour, self.time.minute]])
        self.trainer_params = trainer_params
        #create df with hyper_params
        hps = self.suggest_values()
        hp_names = []
        for hp in hps:
            keys,_ = hp
            hp_names.append(".".join(keys))
        col_names = hp_names + ['pruned','score']
        self.df_hp = pd.DataFrame(columns=col_names)


        class TrainerWithHook(trainer_cls):
            def __init__(self,trial_id,iter,*args,**kwargs):
                self.iter_to_stop = iter
                self.trial_id = trial_id
                super().__init__(*args,**kwargs)


            def build_hooks(self):
                res = super().build_hooks()
                print('sent',self.iter_to_stop,'to hook')
                hook = StopAtIterHook(f"{self.trial_id}_stop_at_{self.iter_to_stop}", self.iter_to_stop)
                res.append(hook)
                return res
        self.trainer_cls = TrainerWithHook

    # parameters end




    def suggest_values(self):
        '''
        generates (possibly random) values for cfg keys.
        output should be list( ( list(str), value) )
        where the list of strings gives the keys identifying the value to be set in the config dict.
        '''
        raise NotImplementedError

    def prune_handling(self,pruned_ids):
        '''
            What to do with the models that has been pruned.
        '''
        pass
#        for trial_id in pruned_ids:
#            shutil.rmtree(self.get_trial_output_dir(trial_id))




    def get_model_name(self,trial_id):
        return f'{self.model_name}_{trial_id}'

    def get_trial_output_dir(self,trial_id):
        return f'{self.output_dir}/trials/{self.get_model_name(trial_id)}_output'

    def load_from_cfg(self,cfg,res,trial_id=-1):
        '''
        load a model specified by cfg and train
        '''
        cfg.SOLVER.MAX_ITER += self.step_chunk_size*res
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = self.trainer_cls(trial_id=trial_id,iter=res * self.step_chunk_size,cfg=cfg,**self.trainer_params)
        trainer.resume_or_load(resume= True)
        return trainer

    def validate(self,cfg_sg,trainer):
      '''
      takes a partially trained model and evaluate it
      '''
      cfg_sg_pred = cfg_sg.clone()
      cfg_sg_pred.MODEL.WEIGHTS = os.path.join(cfg_sg.OUTPUT_DIR, "model_final.pth")
      val_loader = build_detection_test_loader(cfg_sg_pred, self.data_val_name) #ud af loop?
      infe = inference_on_dataset(trainer.model, val_loader, self.evaluator)
      val_to_report = infe[self.task][self.score_name]
      return val_to_report

    def build_cfg(self,trial_id):
        cfg_sg = self.cfg_base.clone()
        vals = []
        for hp in self.suggest_values():
            subdict = cfg_sg
            keys,val = hp
            for key in keys[:-1]:
                subdict = subdict[key.upper()]
            subdict[keys[-1].upper()] = val
            vals.append(val)
        vals.extend([False,-1])
        self.df_hp.loc[trial_id] = vals

        cfg_sg.OUTPUT_DIR = self.get_trial_output_dir(trial_id)
        self.suggested_cfgs.append(cfg_sg)
        return cfg_sg


    def sprint(self,trial_id,res,cfg_sg):
        trainer = self.load_from_cfg(cfg_sg, res,trial_id)
        try:
            trainer.train()
        except hooks.StopFakeExc:
            print("Hyperopt::Stopped per request of hook")
            val_to_report = self.validate(cfg_sg,trainer)
        except (FloatingPointError, ValueError):
            print("Hyperopt::Bad_model")
            val_to_report = 0
        else:
            val_to_report = self.validate(cfg_sg,trainer)
        return val_to_report


    def write_cfg_files(self):
        for trial_id,cfg in enumerate(self.suggested_cfgs):
            yaml = cfg.dump()
            os.makedirs(self.get_trial_output_dir(trial_id),exist_ok=True)
            with open(f"{self.get_trial_output_dir(trial_id)}/cfg.yaml","w+") as fp:
                fp.write(yaml)


    def start(self):
        for i in range(self.pruner.participants):
            self.build_cfg(i)
        self.write_cfg_files()

        id_cur = 0
        done = False
        while not done:
            print("HyperOpt::NOW RUNNING ID,----------------------------------------------------------------------",id_cur)
            print(self.df_hp.loc[id_cur,:])
            cfg = self.suggested_cfgs[id_cur]
            val_to_report = self.sprint(id_cur,self.pruner.get_cur_res(),cfg)
            self.df_hp.loc[id_cur,'score'] = val_to_report
            id_cur, pruned, done = self.pruner.report_and_get_next_trial(val_to_report)
            if pruned:
                self.df_hp.loc[pruned,'pruned'] = True
            self.prune_handling(pruned)

        self.df_hp.to_csv(f'{self.output_dir}/hyperopt_results-{self.time_info_str}.csv')
        return self.df_hp

    def get_result(self):
        return self.pruner.get_best_models()