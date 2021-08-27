import cv2
import numpy as np
from spoleben_train.Tilings import Tilings
from spoleben_train.data_utils import sort_by_prefix,load_masks,load_and_batch_masks,get_small_masks
import os
import shutil

#crop = 200,250,800,1400
#tilings = Tilings((1920,1080),(200,200),(400,400),crop)
base_dir = "/pers_files/spoleben/FRPA_annotering/Automatisering_uge27/Skal_Annoteres"
new_dir ="/pers_files/spoleben/FRPA_annotering/annotations"
os.makedirs(new_dir,exist_ok=False)
subfolder_aug = ["Direct","Tilted","Rotated"]
subfolder_filled = ["15","30","45","60","85","100"]

def batch_masks_in_dir(dir_from,dir_to):
    file_pairs = sort_by_prefix(dir)
    for key, value in file_pairs.items():
        if len(value) > 1:
            fp_ls = [os.path.join(dir_from, name) for name in value[1:] if name.endswith(".png")]
            res = load_and_batch_masks(fp_ls)
            print((res == 255).sum(axis=(1, 2)), (res == 1).sum(axis=(1, 2)))
            print("saving masks of shape", res.shape)
            np.save(os.path.join(dir_to, key) + "_masks.npy", res)
            shutil.copy(os.path.join(dir_from, value[0]), os.path.join(dir_to, value[0]))  # copy

#SCRIPT:
# for direction in subfolder_aug:
#     for fillup in subfolder_filled:
#         dir=os.path.join(base_dir,direction,fillup)
#         batch_masks_to_npy(dir,new_dir)

