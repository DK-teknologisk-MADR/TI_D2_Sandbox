import os , shutil , json
import os.path as path
from copy import deepcopy

import torch.cuda
from numpy.random import choice, randint,uniform

import cv2_utils.cv2_utils
from detectron2_ML.trainers import Trainer_With_Early_Stop
from detectron2.config import get_cfg
from cv2_utils.cv2_utils import *
from detectron2 import model_zoo
from detectron2.data import DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator
from detectron2_ML.pruners import SHA
from detectron2_ML.trainers import TrainerWithMapper,TrainerPeriodicEval
from detectron2.data.detection_utils import transform_instance_annotations,annotations_to_instances

from detectron2_ML.data_utils import get_data_dicts, register_data
from detectron2_ML.transforms import augment_data_with_transform_and_save_as_json
splits = ['train','val']
data_dir_prod = "/pers_files/Combined_final/Filet-10-21/annotated_total" #"/pers_files/Combined_final/Filet-10-21/annotated_530x910"
#production_line data
COCO_dicts_prod = {split: get_data_dicts(data_dir_prod,split) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names_prod = register_data('filet_prod',['train','val'],COCO_dicts_prod,{'thing_classes' : ['filet']}) #register data by str name in D2 api
for i in range(2):
    transforms = [T.HFlipTransform(width=910)]
    for split in splits:
        output = path.join(data_dir_prod +"_aug",split)
        augment_data_with_transform_and_save_as_json(COCO_dicts_prod[split],transforms=transforms,image_size = (530,910),output_dir=output,suffix = f"hflip{i}")
        transforms = [T.VFlipTransform(height=530)]
        augment_data_with_transform_and_save_as_json(COCO_dicts_prod[split],transforms=transforms,image_size = (530,910),output_dir=output,suffix = f"vflip{i}")
        transforms = [T.VFlipTransform(height=530),T.HFlipTransform(width=910)]
        augment_data_with_transform_and_save_as_json(COCO_dicts_prod[split],transforms=transforms,image_size = (530,910),output_dir=output,suffix = f"hvflip{i}")



for data_dict in COCO_dicts_prod['train']:
    polys = [data_dict['annotations'][i]['segmentation'][0] for i in range(len(data_dict['annotations']))]
    polys = [np.array(poly).reshape(-1,2) for poly in polys]
    img = cv2.imread(data_dict['file_name'])
    ol = put_poly_overlays(img,polys)
    checkout_imgs(ol)