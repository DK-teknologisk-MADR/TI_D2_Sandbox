import os , shutil , json
from copy import deepcopy
from numpy.random import choice, randint,uniform
from detectron2_ML.trainers import TI_Trainer,Trainer_With_Early_Stop
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator
from detectron2_ML.pruners import SHA
from detectron2_ML.trainers import TrainerPeriodicEval
from detectron2_ML.hyperoptimization import D2_hyperopt_Base
from numpy import random
from detectron2_ML.data_utils import get_data_dicts, register_data , get_file_pairs
from spoleben_train.data_utils import get_data_dicts_masks,sort_by_prefix

splits = ['train','val']
data_dir = "/pers_files/spoleben/FRPA_annotering/annotations_crop(180,330,820,1450)"
file_pairs = { split : get_file_pairs(data_dir,split,sorted=True) for split in splits }
COCO_dicts = {split: get_data_dicts_masks(data_dir,split,file_pairs[split]) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names = register_data('filet',['train','val'],COCO_dicts,{'thing_classes' : ['spoleben']}) #register data by str name in D2 api
output_dir = f'{data_dir}/output3'



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
    cfg.TEST.EVAL_PERIOD = 50 #set >0 to activate evaluation
    cfg.DATALOADER.NUM_WORKERS = 6 #add more workerss until it gives warnings.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{model_name}.yaml')
    cfg.SOLVER.IMS_PER_BATCH = 3 #maybe more?
    cfg.OUTPUT_DIR = f'{output_dir}/{model_name}_output'
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 90000000
    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.SOLVER.STEPS = [] #cfg.SOLVER.STEPS = [2000,4000] would decay LR by cfg.SOLVER.GAMMA at steps 2000,4000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  #(default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    #might as well make sure output_dir exists
    os.makedirs(f'{output_dir}/{model_name}_output',exist_ok=True)
    return cfg

#example input
model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"
#model_name = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x'
cfg = initialize_base_cfg(model_name)


augmentations = [
          T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
          T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
          T.RandomBrightness(0.9,1.1),
          T.RandomExtent([0.9, 1], [0, 0]),
          T.RandomRotation(angle=[-10, 10], expand=False, center=None, sample_style='range'),
          T.Resize((1024,1024)),

]


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

trainer_params = {'augmentations' : augmentations}
task = 'segm'
evaluator = COCOEvaluator(data_names['val'],("bbox", "segm"), False,cfg.OUTPUT_DIR)

#hyperoptimization object that uses model_dict to use correct model, and get all hyper-parameters.
#optimized after "task" as computed by "evaluator". The pruner is (default) SHA, with passed params pr_params.
#number of trials, are chosen so that the maximum total number of steps does not exceed max_iter.
hyp = D2_hyperopt(model_name,cfg_base=cfg,data_val_name = data_names['val'],trainer_cls=Trainer_With_Early_Stop,task=task,evaluator=evaluator,step_chunk_size=150,output_dir=output_dir,pruner_cls=SHA,max_iter = 100000,trainer_params=trainer_params)
best_models = hyp.start()
#returns pandas object
print(best_models)
