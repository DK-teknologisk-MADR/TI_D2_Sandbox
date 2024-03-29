import gc
import os

import torch.cuda
from detectron2.config import get_cfg
import numpy as np
import pandas as pd
from detectron2_ML.trainers import TI_Trainer,Trainer_With_Early_Stop
from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator
from detectron2_ML import hooks
import detectron2.data.transforms as T
from detectron2_ML.data_utils import get_data_dicts,register_data


pd.set_option('display.max_columns', None)
data_dir = "/pers_files/Combined_final/cropped"
input_dir = f'{data_dir}/output'
model_name_base = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x"
output_dir = os.path.join(data_dir,'output_train')
os.makedirs(output_dir,exist_ok=True)
splits = ['train','val']
#data_dir = "/pers_files/Combined_final/Filet"
COCO_dicts = {split: get_data_dicts(data_dir,split) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names = register_data('filet',['train','val'],COCO_dicts,{'thing_classes' : ['filet']}) #register data by str name in D2 api
model_name = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x"

dfs = []

files = os.listdir(input_dir)
for str in files:
    if str.endswith(".csv"):
        df = pd.read_csv(os.path.join(input_dir,str))
#        df['round']= i
        dfs.append(df)
df = pd.concat(dfs,axis=0,ignore_index=False)
df = df.iloc[:,1:]
df_alive =df.loc[np.logical_not(df['pruned']),:]
df_alive = df_alive.sort_values(by="score")
print(df_alive)
cfgs = []
df_alive['score_final'] = 0


augmentations = [
          T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
          T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
          T.RandomSaturation(intensity_min=0.9,intensity_max=1.1),
          T.RandomContrast(intensity_min=0.8,intensity_max=1.1),
          T.RandomBrightness(0.85,1.15)]
df_alive.to_csv(os.path.join(output_dir,'pd_result_pre.csv'))
#
# class Trainer(TI_Trainer):
#     '''
#     Example of a trainer that applies argumentations at runtime. Argumentations available can be found here:
#     https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html
#     '''
#     def __init__(self,augmentations,**params_to_Trainer):
#         self.augmentations=augmentations
#         self.period_between_evals = cfg.TEST.EVAL_PERIOD
#         self.top_score_achieved = 0
#         super().__init__(**params_to_Trainer)
#     #overwrites default build_train_loader
#     @classmethod
#     def build_train_loader(cls, cfg):
#           mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
#           return build_detection_train_loader(cfg,mapper=mapper)
#
#     @classmethod
#     def build_evaluator(cls,cfg,dataset_name,output_folder=None):
#         if output_folder is None:
#             output_folder = cfg.OUTPUT_DIR
#         return COCOEvaluator(dataset_name, ('bbox', 'segm'), False, output_dir=output_folder)
#
#
#     def build_hooks(self):
#         res = super().build_hooks()
#
#         res.append(hooks.StopByProgressHook(patience=25 * self.period_between_evals, delta_improvement=0.5, score_storage_key='segm/AP', save_name_base="best_model"))
#         return res
#
#     def helper_after_train(self,**kwargs):
#         self.top_score_achieved = self.storage.latest()[f'best_segm/AP']
#
#     def handle_stop(self,**kwargs):
#         self.helper_after_train()
#
#     def handle_else(self,**kwargs):
#         self.helper_after_train()
#
def do_train(augmentations,cfg,eval_period):
    trainer = Trainer_With_Early_Stop(augmentations=augmentations,patience=20*eval_period, cfg=cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()
    return trainer.top_score_achieved[0]

for i in range(1,df_alive.shape[0]):
    eval_period = 250
    model_specs = df_alive.iloc[i,:]
    output_dir = input_dir
    model_name = f"{model_name_base}_{model_specs.name}_output"
    cfg_dir =  os.path.join(data_dir,output_dir,'trials',model_name,'cfg.yaml')
    cfg = get_cfg()
    cfg.merge_from_file(cfg_dir)
    cfg.SOLVER.MAX_ITER = 40000
    cfg.SOLVER.STEPS = (6000,12000,18000,24000,30000,)
    cfg.SOLVER.GAMMA = 0.75
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,"best_model.pth")
    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.MODEL.DEVICE = 'cuda:1'
#    os.makedirs(cfg.OUTPUT_DIR)
    cfg.DATASETS.TEST = ('filet_val',)
    cfgs.append(cfg)
    df_alive.iloc[i,-1] = do_train(augmentations,cfg,eval_period)
    torch.cuda.empty_cache()
    gc.collect()

df_alive.to_csv(os.path.join(output_dir,'pd_result.csv'))

print(df_alive)

#--------------------------------------------
