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

augmentations = [
    RandomCropAndRmPartials(0.3, (450, 450)),
    T.RandomRotation(angle=[-10, 10], expand=False, center=None, sample_style='range'),
    #        T.RandomApply(T.RandomCrop('absolute',(400,400)),prob=0.75),
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
    T.RandomBrightness(0.8, 1.2),
    T.RandomSaturation(0.8, 1.2),
    #          T.Resize((800,800))
]

def gen_val_set(img_dir,save_dir,cycles,augmentations):
    pairs = sort_by_prefix(img_dir)
    aug = T.AugmentationList(augmentations)
    for cycle in range(cycles):
        for front,ls in pairs.items():
            for name in ls:
                if ".jpg" in name:
                    img = cv2.imread(os.path.join(img_dir,name))
                    inp = T.AugInput(image=img)
                    tr = aug(inp)
                    aug_img = tr.apply_image(img)
                    cv2.imwrite(os.path.join(save_dir,name[:-4] + f"aug_nr{cycle}.jpg"),aug_img)

img_dir = "/pers_files/spoleben/spoleben_09_2021/spoleben_not_annotated"
save_dir = "/pers_files/spoleben/spoleben_09_2021/spoleben_augmented"
gen_val_set(img_dir,save_dir=save_dir,cycles = 3,augmentations=augmentations)