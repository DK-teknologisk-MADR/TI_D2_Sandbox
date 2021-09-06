import cv2
import numpy as np
from spoleben_train.Tilings import Tilings
from spoleben_train.data_utils import sort_by_prefix,load_masks,load_and_batch_masks,percentage_of_masks_in_crop,get_small_masks,show_mask
import os

import shutil
img_size = (1080,1920)
crop = 180,330,820,1450
split_size = (240,240)
window_size = (400,400)
crop_mask_th = 0.01
tilings = Tilings(img_size = img_size,split_sizes=split_size,window_size=window_size,crop_size=crop)
crop_sizes = tilings.get_crops()
tilings.get_crops()
dir ="/pers_files/spoleben/FRPA_annotering/annotations"
new_dir = "/pers_files/spoleben/FRPA_annotering/annotations_crop" + str(crop).replace(" ","")
#tilings = Tilings()
for split in ['val']:
    split_dir = os.path.join(dir,split)
    new_split_dir = os.path.join(new_dir,split)
    os.makedirs(new_split_dir, exist_ok=False)
    file_pairs = sort_by_prefix(split_dir)
    for key,values in file_pairs.items():
        for i,crop in enumerate(crop_sizes):
            pic = cv2.imread(os.path.join(split_dir,values[0]))
            crop_pic = pic[crop[0]:crop[2],crop[1]:crop[3]]
            new_pic_name = key + str(i) + ".jpg"
            print(new_pic_name)
            cv2.imwrite(os.path.join(new_split_dir,new_pic_name),crop_pic)
            masks = np.load(os.path.join(split_dir,values[1]))
            print("cropscript: mask shape is",masks.shape,"pic shape is",pic.shape)
            is_large = np.logical_not( get_small_masks(masks,100))
            large_masks = masks[is_large,:,:]
            print("cropscript:there are ",large_masks.shape[0], "large masks in file",key)
            masks_in_crop = percentage_of_masks_in_crop(large_masks,crop)>crop_mask_th
            masks_crop = large_masks[masks_in_crop,crop[0]:crop[2],crop[1]:crop[3]]
            print("cropscript: mask crop shape is",masks_crop.shape)
            print("cropscript: there are ",masks_crop.shape[0], "masks in file",new_pic_name)
            new_mask_name = key + str(i) + ".npy"
            np.save(os.path.join(new_split_dir,new_mask_name),masks_crop)
            print(crop_pic.shape,masks_crop.shape)


#from cv2_utils.cv2_utils import put_mask_overlays()