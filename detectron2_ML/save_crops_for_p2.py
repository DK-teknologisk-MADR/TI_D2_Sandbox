import numpy as np
import torch
from data_and_file_utils.file_utils import get_ending_of_file
import os.path as path
from detectron2_ML.Kpt_Predictor import ModelTester_Aug
from detectron2.utils.visualizer import Visualizer
from cv2_utils.cv2_utils import *
import detectron2.data.transforms as T
from detectron2_ML.transforms import RandomCropAndRmPartials
from torchvision.transforms import ToTensor
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from spoleben_train import Tilings
import os
import cv2

#PARAMS
input_dir = "/pers_files/spoleben/spoleben_09_2021/spoleben_not_annotated"
output_dir = "/pers_files/spoleben/spoleben_09_2021/spoleben_masks"
os.makedirs(path.join(output_dir,'raw'))
os.makedirs(path.join(output_dir,'overview'))
p1_model_dir ="/pers_files/spoleben/spoleben_09_2021/output_22-10/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_226_output"
pic_endings = ['jpg']
crop_size = (450,450)
kpts_out = 3 #doesnt matter
crop_aug = RandomCropAndRmPartials(min_pct_kept=0.6,crop_size=crop_size)
out_size = (180,180)
augmentations = [
          T.RandomBrightness(0.7,1.3),
          T.RandomSaturation(0.7,1.3),
          T.RandomContrast(0.7,1.3)
]
augmentations = T.AugmentationList(augmentations)
tester = ModelTester_Aug(cfg_fp=os.path.join(p1_model_dir, 'cfg.yaml'), chk_fp = os.path.join(p1_model_dir, 'best_model.pth'), skip_phase2=True,  img_size=crop_size, device='cuda:0',record_plots=False)
fps = [path.join(input_dir,file) for file in os.listdir(input_dir) if get_ending_of_file(file) in pic_endings]
for cycle in range(3):
    for fp in fps:
        name = path.basename(fp)
        front = name.split(".")[0]
        img = cv2.imread(fp)
        inp = T.AugInput(image=img)
        crop_tr = crop_aug(inp)
        tr = augmentations(inp)
        tile = crop_tr.apply_image(img)
        aug_img = tr.apply_image(tile)
        masks = tester.phase1(aug_img)[0].to('cpu').numpy().astype(np.uint8)
        large_masks = masks[masks.sum(axis=(1,2))>2300]
        for i,mask in enumerate(large_masks):
            cx, cy = centroid_of_mask_in_xy(mask)
            cx, cy = int(cx), int(cy)
            x0,y0,x1,y1 = cx - out_size[1] // 2 ,cy - out_size[0] // 2, cx + out_size[1] // 2 , cy + out_size[0] // 2
            if x0>0 and x0<out_size[1] and y0>0 and y0<out_size[0]:
                crop_img = tile[cy-out_size[0]//2 : cy+out_size[0]//2,cx-out_size[1]//2 : cx+out_size[1]//2]
                crop_mask = mask[cy-out_size[0]//2 : cy+out_size[0]//2,cx-out_size[1]//2 : cx+out_size[1]//2]
                ol = put_mask_overlays(crop_img, crop_mask, alpha=0.75)
                overview_img = np.hstack([ol,crop_img,    masks_as_color_imgs(crop_mask)*255])
                cv2.imwrite(path.join(output_dir,'raw',front + f"crop{i}_c{cycle}" + "x.jpg"),crop_img)
                cv2.imwrite(path.join(output_dir,'raw',front + f"mask{i}_c{cycle}" + "x.jpg"),crop_mask * 255)
                cv2.imwrite(path.join(output_dir,'overview',front + f"overview{i}_c{cycle}" + "x.jpg"),overview_img)
               # print(fp)
               # checkout_imgs([tile,overview_img])
