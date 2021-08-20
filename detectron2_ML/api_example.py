import os , shutil , json
from copy import deepcopy
from numpy.random import choice, randint,uniform
from trainers import TI_Trainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator
from detectron2_ML.pruners import SHA
from detectron2_ML.trainers import TrainerWithMapper,TrainerPeriodicEval
from hyperoptimization import D2_hyperopt_Base
from numpy import random
from data_utils import get_data_dicts, register_data

splits = ['train','val']
data_dir = "/pers_files/Combined_final/cropped"
COCO_dicts = {split: get_data_dicts(data_dir,split) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names = register_data('filet',['train','val'],COCO_dicts,{'thing_classes' : ['filet']}) #register data by str name in D2 api
output_dir = f'{data_dir}/output_test'
print(data_names)


def initialize_base_cfg(model_name,cfg=None):
    '''
    name of function not important. Sets up the base config for model you want to train.
    SEE MORE AT https://detectron2.readthedocs.io/en/latest/modules/config.html
    '''
    if cfg is None:
        cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f'{model_name}.yaml'))
    cfg.DATASETS.TRAIN = (data_names['train'],)
    cfg.DATASETS.TEST = (data_names['val'],) # Use this with trainer_cls : TrainerPeriodicEval if you want to do validation every #.TEST.EVAL_PERIOD iterations
    cfg.DATASETS.TEST = []
    cfg.TEST.EVAL_PERIOD = 300 #set >0 to activate evaluation
    cfg.DATALOADER.NUM_WORKERS = 1 #add more workerss until it gives warnings.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{model_name}.yaml')
    cfg.SOLVER.IMS_PER_BATCH = 1 #maybe more?
    cfg.OUTPUT_DIR = f'{output_dir}/{model_name}_output'
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = [] #cfg.SOLVER.STEPS = [2000,4000] would decay LR by cfg.SOLVER.GAMMA at steps 2000,4000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  #(default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    #might as well make sure output_dir exists
    os.makedirs(f'{output_dir}/{model_name}_output_test',exist_ok=True)
    return cfg



class D2_hyperopt(D2_hyperopt_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('task is',self.task)


    def suggest_values(self):
        hps = [
            (['solver', 'BASE_LR'], random.uniform(0.0001, 0.0006)),
        ]
        return hps


    def prune_handling(self,pruned_ids):
        for trial_id in pruned_ids:
            shutil.rmtree(self.get_trial_output_dir(trial_id))

#example input
#model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"
model_name = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x'
cfg = initialize_base_cfg(model_name)
task = 'bbox'
evaluator = COCOEvaluator(data_names['val'],("bbox", "segm"), False,cfg.OUTPUT_DIR)

#hyperoptimization object that uses model_dict to use correct model, and get all hyper-parameters.
#optimized after "task" as computed by "evaluator". The pruner is (default) SHA, with passed params pr_params.
#number of trials, are chosen so that the maximum total number of steps does not exceed max_iter.

#hyp = D2_hyperopt(model_name,cfg_base=cfg,data_val_name = data_names['val'],trainer_cls=TI_Trainer,task=task,evaluator=evaluator,step_chunk_size=10,output_dir=output_dir,pruner_cls=SHA,max_iter = 350)
#best_models = hyp.start()
#returns pandas object


#-----------------------------
#example of a training procedure

augmentations = [
          T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
          T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
          T.RandomRotation(angle = [-5,5], expand=True, center=None, sample_style='range'),
          T.RandomBrightness(0.85,1.15)
              ]

trainer = TrainerWithMapper(augmentations = augmentations,cfg=cfg)
trainer.resume_or_load(resume=True)
trainer.train()

#--------------------------------------------

#if argumentations are not neccesary, one does not need subclass
#trainer = TrainerWithMapper(cfg)
#trainer.resume_or_load(resume=True)
#trainer.train()
