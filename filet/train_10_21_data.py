
import os , shutil , json
import os.path as path
from copy import deepcopy

import torch.cuda
from numpy.random import choice, randint,uniform
from detectron2_ML.trainers import Trainer_With_Early_Stop
from detectron2.config import get_cfg

from detectron2 import model_zoo
from detectron2.data import DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator
from detectron2_ML.pruners import SHA
from detectron2_ML.trainers import TrainerWithMapper,TrainerPeriodicEval
from detectron2.data.detection_utils import transform_instance_annotations,annotations_to_instances

from detectron2_ML.data_utils import get_data_dicts, register_data
model_dir = "/pers_files/Combined_final/cropped/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_48_output/fine_tune_10_21"

splits = ['train','val']
data_dir_orig = "/pers_files/Combined_final/Filet"
COCO_dicts_orig = {split: get_data_dicts(data_dir_orig,split) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names_orig = register_data('filet_orig',['train','val'],COCO_dicts_orig,{'thing_classes' : ['filet']}) #register data by str name in D2 api
data_dir_prod = "/pers_files/Combined_final/Filet-10-21/annotated_total" #"/pers_files/Combined_final/Filet-10-21/annotated_530x910"
#production_line data
COCO_dicts_prod = {split: get_data_dicts(data_dir_prod,split) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names_prod = register_data('filet_prod',['train','val'],COCO_dicts_prod,{'thing_classes' : ['filet']}) #register data by str name in D2 api
output_dir = path.join(model_dir,'output')

cfg = get_cfg()
cfg.merge_from_file(path.join(model_dir,'cfg.yaml'))
cfg.DATASETS.TRAIN = (data_names_orig['train'],data_names_prod['train'])
cfg.DATASETS.TEST = (data_names_orig['val'],) # Use this with trainer_cls : TrainerPeriodicEval if you want to do validation every #.TEST.EVAL_PERIOD iterations
cfg.TEST.EVAL_PERIOD = 200 #set >0 to activate evaluation
cfg.DATALOADER.NUM_WORKERS = 6 #add more workerss until it gives warnings.
cfg.MODEL.WEIGHTS = path.join(model_dir,'best_model.pth')
cfg.INPUT.MIN_SIZE_TEST  = 0
cfg.INPUT.MAX_SIZE_TEST = 0
cfg.INPUT.MIN_SIZE_TRAIN  = 0
cfg.INPUT.MAX_SIZE_TRAIN = 0
cfg.SOLVER.IMS_PER_BATCH = 6 #maybe more?
cfg.OUTPUT_DIR = output_dir
cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR / 5
cfg.SOLVER.MAX_ITER = 500000//20
cfg.SOLVER.STEPS = [] #cfg.SOLVER.STEPS = [2000,4000] would decay LR by cfg.SOLVER.GAMMA at steps 2000,4000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
#might as well make sure output_dir exists
os.makedirs(output_dir)
with open(path.join(output_dir,"cfg.yaml"), "w+") as f:
  f.write(cfg.dump())   # save config to file

str = cfg.dump()
augmentations = [
          T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
          T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
          T.RandomBrightness(0.85,1.15),
          T.RandomSaturation(0.85,1.15),
          T.RandomContrast(0.85,1.1),
]



trainer = Trainer_With_Early_Stop(augmentations = augmentations,patience = 20000,cfg=cfg)
trainer.resume_or_load(resume=True)
trainer.train()
