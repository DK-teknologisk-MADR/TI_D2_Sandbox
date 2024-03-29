import os , shutil , json,cv2

import detectron2.utils.visualizer
from detectron2.utils.visualizer import Visualizer
import numpy as np
from copy import deepcopy
from numpy.random import choice, randint,uniform
from cv2_utils.colors import RGB_TO_COLOR_DICT
from cv2_utils.cv2_utils import *
from detectron2_ML.trainers import TI_Trainer,Trainer_With_Early_Stop
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator
from detectron2_ML.pruners import SHA
from detectron2_ML.trainers import TrainerPeriodicEval
from detectron2_ML.hyperoptimization import D2_hyperopt_Base
from datetime import datetime
from numpy import random
from detectron2_ML.data_utils import get_data_dicts, register_data , get_file_pairs,sort_by_prefix
from spoleben_train.data_utils import get_data_dicts_masks
from detectron2_ML.transforms import RemoveSmallest , CropAndRmPartials,RandomCropAndRmPartials,RandomChoiceAugmentation
splits = ['train','val']
data_dir = '/pers_files/spoleben/spoleben_09_2021/spoleben_for_training'
file_pairs = { split : sort_by_prefix(os.path.join(data_dir,split)) for split in splits }
#file_pairs = { split : get_file_pairs(data_dir,split,sorted=True) for split in splits }
COCO_dicts = {split: get_data_dicts_masks(data_dir,split,file_pairs[split]) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names = register_data('filet',splits,COCO_dicts,{'thing_classes' : ['spoleben']}) #register data by str name in D2 api
output_dir = f'/pers_files/spoleben/spoleben_09_2021/output_{datetime.now().day}-{datetime.now().month}'
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
    cfg.TEST.EVAL_PERIOD = 200 #set >0 to activate evaluation
    cfg.DATALOADER.NUM_WORKERS = 6 #add more workerss until it gives warnings.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{model_name}.yaml')
    cfg.SOLVER.IMS_PER_BATCH = 4 #maybe more?
    cfg.OUTPUT_DIR = f'{output_dir}/{model_name}_output_eval'
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 90000000
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.INPUT.MIN_SIZE_TEST = 450
    cfg.INPUT.MAX_SIZE_TEST = 450
    cfg.INPUT.MIN_SIZE_TRAIN = (450,450) #min size train SKAL være tuple ved prediction af en eller anden årsag.
    cfg.INPUT.MAX_SIZE_TRAIN = 450
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / cfg.SOLVER.WARMUP_ITERS
    cfg.SOLVER.STEPS = [] #cfg.SOLVER.STEPS = [2000,4000] would decay LR by cfg.SOLVER.GAMMA at steps 2000,4000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  #(default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    #might as well make sure output_dir exists
    os.makedirs(f'{output_dir}/{model_name}_output',exist_ok=True)
    return cfg

#example input
#model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"
model_name = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x'
cfg = initialize_base_cfg(model_name)
crop_aug_ls = [RandomCropAndRmPartials(0.25,(410,410)),RandomCropAndRmPartials(0.25,(425,425)),RandomCropAndRmPartials(0.3,(450,450)),RandomCropAndRmPartials(0.3,(475,475)),RandomCropAndRmPartials(0.3,(490,490))]
crop_aug = RandomChoiceAugmentation(crop_aug_ls)
augmentations = [
          crop_aug,
          T.Resize((450,450)),
          T.RandomRotation(angle=[-10, 10], expand=False, center=None, sample_style='range'),
          T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
          T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
          T.RandomBrightness(0.75,1.25),
          T.RandomSaturation(0.75,1.25),
          T.RandomContrast(0.75,1.1)
]


#img_basename = 'kinect_20210916_102523_color__ID193'
#img = cv2.imread(os.path.join(data_dir,'',img_basename + ".jpg"))
#masks = np.load(os.path.join(data_dir,'',img_basename + "_masks.npy"))
#aug = T.AugmentationList(augmentations)
#inp = T.AugInput(image=img)
#tr = aug(inp)
#a = tr.apply_image(img)
#checkout_imgs(a)
#masks = [ tr.apply_segmentation(mask.astype('uint8') * 255) for mask in masks]

#checkout_imgs(masks[0])
#colors = list(RGB_TO_COLOR_DICT.keys()).copy()
#np.random.shuffle(colors)
#mask_new = put_mask_overlays(a,masks,colors)

#checkout_imgs(mask_new)
# pairs = sort_by_prefix(os.path.join(data_dir,'train'))
# for front,pair in pairs.items():
#     jpg,npy = pair
#     img = cv2.imread(os.path.join(data_dir,'train',jpg))
#     masks = np.load(os.path.join(data_dir,'train',npy))
#     img_and_mask = put_mask_overlays(img,masks,colors=[(220,120,0),(0,220,120),(155,155,120),(155,225,40),(70,30,225)])
#     checkout_imgs({front:img_and_mask})
# #augmentations = [RandomCropAndRmPartials(0.55,(450,450))]
# img = cv2.imread(os.path.join(data_dir,'train','robotcell_2021-06-30-14-35-35_all_00000_cam_2_color.jpg'))
# masks = np.load(os.path.join(data_dir,'train','robotcell_2021-06-30-14-35-35_all_00000_cam_2_color_masks.npy'))
# img_and_mask = put_mask_overlays(,masks,colors=[(220,120,0),(0,220,120),(155,155,120),(155,225,40),(70,30,225)])
#
# checkout_imgs(img_and_mask)
# aug = T.AugmentationList(augmentations)
# inp = T.AugInput(image=img)
# tr = aug(inp)
# print(tr)
# a = tr.apply_image(img)
# masks_aug = [tr.apply_segmentation(mask.astype('uint8') * 255) for mask in masks]
# mask_new = put_mask_overlays(a,masks_aug,colors=[(220,120,0),(0,220,120),(155,155,120),(155,225,40),(70,30,225)])
# checkout_imgs([img_and_mask,a,mask_new])



# trs = CropAndRmPartials(0.5,x0=320,y0=200,w=400,h=400,orig_w=640,orig_h=1120)
# for a_mask in masks:
#     mask_crop = trs.apply_segmentation(a_mask.copy())
#     mask_crop = mask_crop.astype('uint8') * 255
#     mask_crop_orig = trs_crop.apply_segmentation(a_mask).astype('uint8') * 255
#     checkout_imgs([img_crop,mask_crop,a_mask.astype('uint8')*255,mask_crop_orig])


class D2_Hyperopt_Spoleben(D2_hyperopt_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('task is',self.task)

    def suggest_helper_size(self):
        all_sizes = [32,64,128]
        all_sizes.extend([128 + 64 * i for i in range(1,7)])
        nr = np.random.randint(2,6)
        ls = [np.sort(np.random.choice(all_sizes, nr,replace=False)).tolist()]

        return ls

    def suggest_values(self):
        hps = [
#            (['model', 'anchor_generator', 'sizes'], self.suggest_helper_size()),
            (['SOLVER','MOMENTUM'],np.random.uniform(0.85,0.95)),
            (['SOLVER','BASE_LR'],float(random.uniform(0.1,2) * random.choice([0.001,0.0001]))),
#            (['model', 'anchor_generator', 'aspect_ratios'], random.choice([[0.75,1.0, 1.5], [0.5, 1.0, 2.0], [1.0]])),
           (['MODEL','RPN','NMS_THRESH'],random.uniform(0.55, 0.85)),
            (['MODEL','RPN','POST_NMS_TOPK_TRAIN'], random.randint(500,1000)),
        ]
        return hps

    def prune_handling(self,pruned_ids):
        for trial_id in pruned_ids:
            shutil.rmtree(self.get_trial_output_dir(trial_id))


task = 'segm'
score_name = 'AP'
evaluator = COCOEvaluator(data_names['val'],("segm",), False,cfg.OUTPUT_DIR)
#evaluator = MeatPickEvaluator(COCO_dicts[''],top_n_ious=3)

#CropAndRmPartials(partial_crop_pct=0.5)
#hyperoptimization object that uses model_dict to use correct model, and get all hyper-parameters.
#optimized after "task" as computed by "evaluator". The pruner is (default) SHA, with passed params pr_params.
#number of trials, are chosen so that the maximum total number of steps does not exceed max_iter.
hyp = D2_Hyperopt_Spoleben(model_name,cfg_base=cfg,data_val_name = data_names['val'],task=task,evaluator=evaluator,step_chunk_size=498,output_dir=output_dir,pruner_cls=SHA,max_iter = 1000000,trainer_params=trainer_params,pr_params={'factor' : 4, 'topK' : 1})
best_models = hyp.start()
#returns pandas object
print(best_models) 
