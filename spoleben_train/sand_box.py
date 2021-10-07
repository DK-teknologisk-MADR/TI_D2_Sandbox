import os , shutil , json,cv2
from detectron2.engine import DefaultPredictor
import detectron2.utils.visualizer
from detectron2.utils.visualizer import Visualizer
import numpy as np
from time import time
from copy import deepcopy
from torchvision.transforms import ColorJitter,RandomAffine,Normalize,ToTensor
from torchvision.transforms.functional import adjust_contrast,adjust_brightness,affine,_get_inverse_affine_matrix
from numpy.random import choice, randint,uniform
from cv2_utils.colors import RGB_TO_COLOR_DICT
from cv2_utils.cv2_utils import *
from detectron2_ML.trainers import TI_Trainer,Trainer_With_Early_Stop
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch
from detectron2_ML.evaluators import Consistency_Evaluator
from detectron2.data import DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator
from detectron2_ML.pruners import SHA
from detectron2_ML.trainers import TrainerPeriodicEval
from detectron2_ML.hyperoptimization import D2_hyperopt_Base
from numpy import random
from detectron2_ML.data_utils import get_data_dicts, register_data , get_file_pairs,sort_by_prefix
from spoleben_train.data_utils import get_data_dicts_masks
from detectron2_ML.transforms import RemoveSmallest , CropAndRmPartials,RandomCropAndRmPartials
from detectron2_ML.evaluators import MeatPickEvaluator
splits = ['']
data_dir = '/pers_files/spoleben/spoleben_09_2021/spoleben_not_annotated_aug' #"/pers_files/spoleben/FRPA_annotering/annotations_crop(180,330,820,1450)"
file_pairs = { split : sort_by_prefix(os.path.join(data_dir,split)) for split in splits }
#file_pairs = { split : get_file_pairs(data_dir,split,sorted=True) for split in splits }
COCO_dicts = {split: get_data_dicts_masks(data_dir,split,file_pairs[split]) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names = register_data('filet',splits,COCO_dicts,{'thing_classes' : ['spoleben']}) #register data by str name in D2 api
output_dir = f'/pers_files/spoleben/spoleben_09_2021/output_test2'
data = COCO_dicts[""]

def initialize_base_cfg(model_name,cfg=None):
    '''
    name of function not important. Sets up the base config for model you want to train.
    SEE MORE AT https://detectron2.readthedocs.io/en/latest/modules/config.html
    '''
    if cfg is None:
        cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f'{model_name}.yaml'))
    cfg.DATASETS.TRAIN = (data_names[''],)
    cfg.DATASETS.TEST = (data_names[''],) # Use this with trainer_cls : TrainerPeriodicEval if you want to do validation every #.TEST.EVAL_PERIOD iterations
    cfg.TEST.EVAL_PERIOD = 200 #set >0 to activate evaluation
    cfg.DATALOADER.NUM_WORKERS = 6 #add more workerss until it gives warnings.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{model_name}.yaml')
    cfg.SOLVER.IMS_PER_BATCH = 2 #maybe more?
    cfg.OUTPUT_DIR = f'{output_dir}/{model_name}_output_eval'
    cfg.SOLVER.BASE_LR = 0.000
    cfg.SOLVER.MAX_ITER = 90000000
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.INPUT.MIN_SIZE_TEST = 450
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / cfg.SOLVER.WARMUP_ITERS
    cfg.SOLVER.STEPS = [] #cfg.SOLVER.STEPS = [2000,4000] would decay LR by cfg.SOLVER.GAMMA at steps 2000,4000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  #(default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    #might as well make sure output_dir exists
    os.makedirs(f'{output_dir}/{model_name}_output',exist_ok=False)
    return cfg




#FOR TESTING#
cfg = get_cfg()

cfg.merge_from_file('/pers_files/spoleben/spoleben_09_2021/output_test/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_6_output/cfg.yaml')
cfg.OUTPUT_DIR = '/pers_files/spoleben/spoleben_09_2021/output_test/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_6_output'
cfg.MODEL.WEIGHTS = '/pers_files/spoleben/spoleben_09_2021/output_test/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_6_output/best_model.pth'
cfg.INPUT.MIN_SIZE_TEST=450

data_ls = deepcopy(COCO_dicts[''])
for data in data_ls:
    data['image'] = torch.tensor(cv2.imread(data['file_name'])).permute(2,0,1)
print("PRINTING DATA",data)
#checkout_imgs(tensor_pic_to_imshow_np(data['image']))
eval = Consistency_Evaluator(predictor_cfg=cfg,coco_dicts_ls=data_ls,top_n_ious=10,img_size=(650,1400),device='cuda:0',min_size_incon=3000)
eval.process(data_ls)
print(eval.evaluate())
#checkout_imgs(tensor_pic_to_imshow_np(data['image']))
#eval = Consistency_Evaluator(predictor_cfg=cfg,coco_dicts_ls=data,top_n_ious=10,img_size=(650,1400),device='cuda:0',min_size_incon=3000)

#eval.process([data])
#print(eval.evaluate())

#
# cfg = get_cfg()
#
# cfg.merge_from_file('/pers_files/spoleben/spoleben_09_2021/output_test/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_6_output/cfg.yaml')
# cfg.OUTPUT_DIR = '/pers_files/spoleben/spoleben_09_2021/output_test/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_6_output'
# cfg.MODEL.WEIGHTS = '/pers_files/spoleben/spoleben_09_2021/output_test/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_6_output/best_model.pth'
# cfg.INPUT.MIN_SIZE_TEST=450
# pred = DefaultPredictor(cfg)
# aug_nr = 4
# angles = [7,-7,7,-7]
# translate = [(37,41),(43,-49),(-45,39),(-39,45)]
# assert len(angles) == aug_nr and len(translate) == aug_nr
# img = cv2.imread(data[0]['file_name'])
# tr_ts = ToTensor()
# img_ts = tr_ts(img)
# img_ts = img_ts.expand(size=(1 + aug_nr,3,650,1400)).clone()
# with torch.no_grad():
#     start = time()
#     img_ts.to('cuda:0')
#     img_ts[1:3,] = adjust_brightness(img_ts[1:3,], brightness_factor = 0.7)
#     img_ts[4:,] = adjust_brightness(img_ts[4:,], brightness_factor = 1.3)
#     img_ts[1,] = adjust_contrast(img_ts[2,], contrast_factor = 0.7)
#     img_ts[4,] = adjust_contrast(img_ts[4,], contrast_factor = 1.3)
#     for i in range(aug_nr):
#         img_ts[1+i] = affine(img_ts[1 + i], angle = angles[i],translate = translate[i],scale=1,shear=[0,0])
#     img_ts_ls = img_ts.split(1)
#     img_ts_dict = [{'image' : img.squeeze(0) * 255} for img in img_ts_ls]
#     output = pred.model(img_ts_dict)
#     pred_masks = [output_dict['instances'].pred_masks for output_dict in output]
#     pred_masks[1].shape
#     for i in range(aug_nr):
#         pred_masks[1+i] = affine(pred_masks[1 + i].unsqueeze(1),angle = 0,translate = (- translate[i][0],- translate[i][1] ),scale = 1,shear = [0 , 0 ]).squeeze_(1)
#     for i in range(aug_nr):
#         pred_masks[1+i] = affine(pred_masks[1 + i].unsqueeze(1),angle = -angles[i],translate=(0,0),scale = 1,shear = [0, 0]).squeeze_(1)
#     torch.cuda.synchronize()
#     end = time()
# print(end-start)
# masks_cpu = [mask.to('cpu').numpy() for mask in pred_masks]
# img = tensor_pic_to_imshow_np(img_ts[0])
# over = put_mask_overlays(img,masks_cpu[4])

#    masks = pred(img)['instances'].pred_masks.to('cpu').numpy()



