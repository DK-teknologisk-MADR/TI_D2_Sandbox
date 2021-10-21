import json
import yaml
import cv2
import os
import os.path as path
import random
import re
from cv2_utils.cv2_utils import *
import cv2_utils.cv2_utils
from detectron2_ML.data_utils import get_file_pairs, get_data_dicts
import detectron2.data.transforms as T
import torch
import numpy as np
from PIL import Image, ImageDraw

import json
import numpy as np
from pycocotools import mask
from skimage import measure

from detectron2_ML.predictors import ModelTester
data_dir = '/pers_files/Combined_final/Filet-10-21/cropped_1024x1024'
out ='/pers_files/Combined_final/Filet-10-21/annotated_530x910'
model_dir = '/pers_files/Combined_final/cropped/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_48_output/'
cfg_fp = path.join(model_dir,'cfg.yaml')
chk_fp = path.join(model_dir,'best_model.pth')

with open(path.join(data_dir,'sample.json')) as fp:
    data_dict_orig = json.load(fp)
data_dict_orig['shapes'] = []
pictures = [name for name in os.listdir(data_dir) if name.endswith(".jpg") or name.endswith(".jpeg")]
x0, y0= 50, 250
h,w = 530,910
x1,y1 = x0 + w,y0 + h
tr = T.CropTransform(x0=x0, y0=y0, w=w, h=h)
# np.random.shuffle(pictures)
# for name in pictures[:10]:
# #    name = pictures[0]
#     fp = path.join(data_dir,name)
#     img = cv2.imread(fp)
#
# #    x0,y0,x1,y1 = (900,500,800+910,600+530)
#
#     img_crop = tr.apply_image(img)
# #    cv2.imshow("win",img)
#     cv2.imshow("wi",img_crop)
#
#     cv2.waitKey()
#     cv2.destroyAllWindows()
label_name = 'filet'
shapes_dict = {'line_color': None, 'fill_color': None, 'label': 'filet', 'shape_type': 'polygon', 'points': None, 'flags': {}, 'group_id': None}
model = ModelTester(cfg_fp=cfg_fp,chk_fp=chk_fp)
for name in pictures:
    fp = path.join(data_dir,name)
    img_orig = cv2.imread(fp)
    img = img_orig
    img = tr.apply_image(img)
    output = model(img)['instances']
    good_masks = output.scores>0.5
    masks = model(img)['instances'][good_masks].pred_masks.to('cpu').numpy().astype(np.uint8)
    data_dict = data_dict_orig.copy()
    data_dict['imagePath'] = name
    data_dict['imageHeight'] = img.shape[0]
    data_dict['imageWidth'] = img.shape[1]
    data_dict['shapes'] = []
    print( len(data_dict['shapes']) )
    print(img.shape)
    print("masks",masks.shape)
    for a_mask in masks:
        ground_truth_binary_mask = a_mask
        fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        #ground_truth_area = mask.area(encoded_ground_truth)
        #ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = measure.find_contours(ground_truth_binary_mask,.5)
        segmentations = []
        print("so many contours by mask", len(contours))
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel()
            segmentation = segmentation.reshape(-1,2)
            segmentation = segmentation.astype(np.float)
            point_ls = segmentation[:-1][::15].tolist()
            new_shape = shapes_dict.copy()
            new_shape['points'] = point_ls
            data_dict['shapes'].append(new_shape)
    cv2.imwrite(path.join(out,name),img)
    with open(path.join(out,name.split(".")[0] + ".json"),'w+') as fp:
        json.dump(data_dict,fp)

#overlay = put_circle_overlays(img,segmentation[:-1][::15])
#checkout_imgs(overlay)
#    annotation["segmentation"].append(segmentation)

#print(json.dumps(annotation, indent=4))
