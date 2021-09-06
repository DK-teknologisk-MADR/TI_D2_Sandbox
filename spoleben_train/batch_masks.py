import cv2
from cv2_utils.cv2_utils import put_mask_overlays,checkout_imgs
import numpy as np
from spoleben_train.Tilings import Tilings
from spoleben_train.data_utils import sort_by_prefix,load_masks,load_and_batch_masks,get_small_masks
import os
import shutil

#crop = 200,250,800,1400
#tilings = Tilings((1920,1080),(200,200),(400,400),crop)
base_dir = "/pers_files/spoleben/FRPA_annotering/Automatisering_uge27/Skal_Annoteres"
new_dir ="/pers_files/spoleben/FRPA_annotering/annotations_case"
os.makedirs(new_dir,exist_ok=False)
subfolder_aug = ["Direct","Tilted","Rotated"]
subfolder_filled = ["15","30","45","60","85","100"]

def batch_masks_in_dir(dir_from,dir_to,cropx0y0x1y1 = None):
    file_pairs = sort_by_prefix(dir_from)
    print(file_pairs)
    for key, value in file_pairs.items():
        if len(value) > 1:
            img = cv2.imread(os.path.join(dir_from, value[0]))
            fp_ls = [os.path.join(dir_from, name) for name in value[1:] if name.endswith(".png")]
            res = load_and_batch_masks(fp_ls)
            if cropx0y0x1y1 is not None:
                x0,y0,x1,y1 = cropx0y0x1y1
                res = res[:,y0:y1,x0:x1]
                img = img[y0:y1,x0:x1,:]
            print((res == 1).sum(axis=(1, 2)))
            print("ARE THERE ANY EMPTIES:",np.logical_not(np.any((res == 1).sum(axis=(1, 2)))))
            print("saving masks of shape", res.shape)

#            overlay = put_mask_overlays(img,res,colors=[(220,120,0),(0,220,120),(155,155,120),(200,185,210)])PLOT
#            checkout_imgs(overlay) PLOT
            np.save(os.path.join(dir_to, key) + "_masks.npy", res)
            cv2.imwrite(os.path.join(dir_to, value[0]),img)  # copy

#batch_masks_in_dir(base_dir,new_dir,(820,1450,180,330))
#SCRIPT:
for direction in subfolder_aug:
     for fillup in subfolder_filled:
         dir=os.path.join(base_dir,direction,fillup)
         batch_masks_in_dir(dir,new_dir,(330,180,1450,820))

