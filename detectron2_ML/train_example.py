import random
import os , shutil
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2_ML.hyperoptimization import D2_hyperopt_Base
from numpy import random
from detectron2_ML.transforms import RandomChoiceAugmentation
from detectron2_ML.trainers import Trainer_With_Early_Stop
from detectron2_ML.data_utils import get_data_dicts, register_data
import detectron2.data.transforms as T
from datetime import datetime
splits = ['train','val']
data_dir = "/pers_files/Combined_final/cropped"
orig_dicts = { split: get_data_dicts(data_dir,split) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_dir_prod = "/pers_files/Combined_final/Filet-10-21/annotated_total_aug" #"/pers_files/Combined_final/Filet-10-21/annotated_530x910"
#production_line data
COCO_dicts_prod = {split: get_data_dicts(data_dir_prod,split) for split in splits } #converting TI-annotation of pictures to COCO annotations.
for split in splits:
    orig_dicts[split].extend(COCO_dicts_prod[split])
data_names = register_data('filet',['train','val'],orig_dicts,{'thing_classes' : ['filet']}) #register data by str name in D2 api


#model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"
model_name = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x'
output_dir = f'{data_dir}_output_{datetime.now().day}-{datetime.now().month}'

rotationaug = RandomChoiceAugmentation([          T.RandomRotation(angle=[-5, 5], expand=False, center=None, sample_style='range'),
                                                  T.RandomRotation(angle=[-5, 5], expand=True, center=None, sample_style='range')])
augmentations = [
          rotationaug,
          T.Resize((530,910)),
          T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
          T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
          T.RandomBrightness(0.85,1.15),
          T.RandomSaturation(0.85,1.15),
          T.RandomContrast(0.80,1.2)
]

def initialize_base_cfg(model_name,output_dir,cfg=None):
    '''
    setup base configuration of model SEE MORE AT https://detectron2.readthedocs.io/en/latest/modules/config.html
    '''
    if cfg is None:
        cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f'{model_name}.yaml'))
    cfg.DATASETS.TRAIN = (data_names['train'],)
    cfg.DATASETS.TEST = (data_names['val'],)
    cfg.TEST.EVAL_PERIOD = 200 #set >0 to activate evaluation
    cfg.DATALOADER.NUM_WORKERS = 6 #add more workerss until it gives warnings.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{model_name}.yaml')
    cfg.SOLVER.IMS_PER_BATCH = 6 #maybe more?
    cfg.OUTPUT_DIR = f'{output_dir}/{model_name}_output'
    cfg.INPUT.MIN_SIZE_TEST = 530
    cfg.INPUT.MAX_SIZE_TEST = 910
    cfg.INPUT.MIN_SIZE_TRAIN = (530,) #min size train SKAL være tuple ved prediction af en eller anden årsag.
    cfg.INPUT.MAX_SIZE_TRAIN = 910
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / cfg.SOLVER.WARMUP_ITERS
    os.makedirs(f'{output_dir}/{model_name}_output',exist_ok=True)
    cfg.SOLVER.MAX_ITER = 1000000
    cfg.SOLVER.STEPS = [] #cfg.SOLVER.STEPS = [2000,4000] would decay LR by cfg.SOLVER.GAMMA at steps 2000,4000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  #(default: 512)t
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE='cuda:1'
    return cfg


cfg = initialize_base_cfg(model_name,output_dir)

task = 'segm'
evaluator = COCOEvaluator(data_names['val'],("segm",), False,cfg.OUTPUT_DIR)
print(data_names[splits[0]],)
print(data_names[splits[1]],)


trainer = Trainer_With_Early_Stop(augmentations = augmentations,patience = 20000,cfg=cfg)
trainer.resume_or_load(resume=True)
trainer.train()