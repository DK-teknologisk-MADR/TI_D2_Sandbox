import cv2
from cv2_utils.cv2_utils import put_mask_overlays,checkout_imgs
import numpy as np
from spoleben_train.Tilings import Tilings
from spoleben_train.data_utils import sort_by_prefix,load_masks,load_and_batch_masks,get_small_masks
from spoleben_train.batch_masks import batch_masks_in_dir,black_out_masks_in_dir,visualize_masks
import os
import shutil

'''
This script takes output of "image annotator" tool ( the picture, and a bunch of .png masks) bundle them together in a numpy mask, and black out the parts of the picture in the "not-annotated" mask.
Requirement: Remember to set the "mask_keyword" to distinquish between which masks are annotation-masks, and which mask is the mask to black out picture.
'''



base_dir = "/pers_files/spoleben/spoleben_09_2021/spoleben_annotated"   # "/pers_files/spoleben/FRPA_annotering/annotations" #
new_dir = "/pers_files/spoleben/spoleben_09_2021/spoleben_batched" #= "/pers_files/spoleben/FRPA_annotering/annotations_crop" + str(crop).replace(" ","")

batch_masks_in_dir(base_dir,new_dir,mask_keywords=["mask_mask","mask_stykke"])
black_out_masks_in_dir(base_dir,new_dir,mask_keywords=["not_annotated",'error'])
visualize_masks(dir=new_dir)

