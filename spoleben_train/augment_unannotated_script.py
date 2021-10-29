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
from numpy import random
from detectron2_ML.data_utils import get_data_dicts, register_data , get_file_pairs,sort_by_prefix
from detectron2_ML.transforms import RemoveSmallest , CropAndRmPartials,RandomCropAndRmPartials
import os.path as path
augmentations = [
          RandomCropAndRmPartials(0.3,(450,450)),
 #         T.RandomRotation(angle=[-10, 10], expand=False, center=None, sample_style='range'),
 #        T.RandomApply(T.RandomCrop('absolute',(400,400)),prob=0.75),
          T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
          T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
          T.RandomBrightness(0.7,1.3),
          T.RandomSaturation(0.7,1.3),
          T.RandomContrast(0.9,1.1)
]


def gen_val_set(img_dir,save_dir,cycles,augmentations,include_annotations = True):
    pairs = sort_by_prefix(img_dir)
    os.makedirs(save_dir,exist_ok=True)
    aug = T.AugmentationList(augmentations)
    for cycle in range(cycles):
        for front,ls in pairs.items():
            jpg_name, npy_name = None,None
            for name in ls:
                if ".jpg" in name:
                    jpg_name = name
                elif ".npy" in name:
                    npy_name = name
            print(jpg_name,npy_name)
            if jpg_name is not None and npy_name is not None:
                img = cv2.imread(path.join(img_dir,jpg_name))
                inp = T.AugInput(image=img)
                tr = aug(inp)
                aug_img = tr.apply_image(img)
                masks = np.load(path.join(img_dir,npy_name))
                print(masks.shape)
                mask_ls = []
                for mask in masks:
                    new_mask = tr.apply_segmentation(mask)
                    mask_ls.append(new_mask)
                    #print(img.shape,aug_img.shape,mask.shape,new_mask.shape)
                    ol = put_mask_overlays(img=img,masks = [mask])
#                    new_ol = put_mask_overlays(aug_img,[new_mask])
                #    checkout_imgs([ol,new_ol])
                new_masks = np.array(mask_ls)
                new_ol = put_mask_overlays(aug_img, new_masks)
                np.save(path.join(save_dir,npy_name[:-9] + f"aug_nr{cycle}x_masks.npy"),new_masks)
                cv2.imwrite(os.path.join(save_dir,jpg_name[:-4] + f"_aug_nr{cycle}x.jpg"),aug_img)
            elif ".json" in name:
                raise ValueError("havent implemented for json and polygon annotations yet")
img_dir = "/pers_files/spoleben/spoleben_09_2021/spoleben_batched/val"
#img_dir = "/pers_files/spoleben/spoleben_09_2021/spoleben_not_annotated"
save_dir = "/pers_files/spoleben/spoleben_09_2021/spoleben_for_training/val"
gen_val_set(img_dir=img_dir,save_dir=save_dir,cycles = 100,augmentations=augmentations)