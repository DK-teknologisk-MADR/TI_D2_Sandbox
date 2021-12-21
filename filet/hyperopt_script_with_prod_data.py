import random
import os , shutil
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2_ML.hyperoptimization import D2_hyperopt_Base
from numpy import random
from copy import deepcopy
from detectron2_ML.transforms import augment_data_with_transform_and_save_as_json,RandomChoiceAugmentation
from detectron2_ML.trainers import Trainer_With_Early_Stop,Hyper_Trainer
from detectron2_ML.data_utils import get_data_dicts, register_data
import detectron2.data.transforms as T
from datetime import datetime
from detectron2.data.detection_utils import transform_instance_annotations
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
class D2_hyperopt(D2_hyperopt_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('task is',self.task)

    def suggest_helper(self,i):
        if i==0:
            return [[64, 128, 256, 512]]
        elif i==1:
            return [[128, 256, 512]]
        else:
            raise ValueError

    def suggest_values(self):
        hps = [
            (['model', 'anchor_generator', 'sizes'],self.suggest_helper(random.randint(0,2))),
            (['model', 'anchor_generator', 'aspect_ratios'], random.choice([[0.5, 1.0, 2.0], [0.25, 0.5, 1.0, 2.0],[0.25,0.5, 1.0]])),
            (['solver', 'BASE_LR'], random.uniform(0.0005, 0.005)),
            (['model', 'roi_heads', 'batch_size_per_image'], int(random.choice([128, 256, 512]))),
            (['MODEL', 'RPN', 'NMS_THRESH'], random.uniform(0.55, 0.85)),
            (['MODEL', 'RPN', 'POST_NMS_TOPK_TRAIN'], random.randint(500, 1000)),
        ]
        return hps

    def prune_handling(self,pruned_ids):
        for trial_id in pruned_ids:
            shutil.rmtree(self.get_trial_output_dir(trial_id))

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
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  #(default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE='cuda:1'
    return cfg


cfg = initialize_base_cfg(model_name,output_dir)
print("------------------------------------------")
print("------------------------------------------")
print("------------------------------------------")
print("------------------------------------------")

print(cfg.SOLVER.WARMUP_ITERS)
print("------------------------------------------")
print("------------------------------------------")
print("------------------------------------------")
print("------------------------------------------")
print("------------------------------------------")

task = 'segm'
evaluator = COCOEvaluator(data_names['val'],("segm",), False,cfg.OUTPUT_DIR)
print(data_names['train'],)
print((data_names['val'],))

round1 = D2_hyperopt(model_name,cfg,data_val_name=data_names['val'],task=task,evaluator=evaluator,output_dir=output_dir,step_chunk_size=600,max_iter=1000000,pr_params={'factor' : 3,'topK' : 1},trainer_cls=Hyper_Trainer,trainer_params = {'augmentations' : augmentations})
res1 = round1.start()
#output_dir2 = f"{data_dir}/output2"
#cfg = initialize_base_cfg(model_name,output_dir2)

#round2 = D2_hyperopt(model_name,cfg,data_val_name=data_names['val'],task=task,evaluator=evaluator,output_dir=output_dir2,step_chunk_size=300,max_iter=150000,pr_params={'factor' : 3, 'topK' : 3})
#res2 = round2.start()
print(res1)
#print(res2)
