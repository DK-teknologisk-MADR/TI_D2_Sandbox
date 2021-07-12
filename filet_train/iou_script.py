import re
from pytorch_ML import get_ious, find_best_iou
import torch
import json
from copy import deepcopy
from detectron2_ML import data_utils
import cv2
from detectron2_ML.data_utils import get_file_pairs
import os
from skimage.draw import polygon2mask
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
torch.cuda.device(1)
from filet_train.Filet_kpt_Predictor import Filet_ModelTester

def crop_out_seg (img,cnt,pads = (0,0),crop_dim=(256+512,256+512)):
    rect = cv2.minAreaRect(cnt)
    points = cv2.boxPoints(rect)
  #  input_img = cv2.circle(input_img, tuple(points[0].astype('int')), 5, (90,224,185), 3)
    cX,cY = points.mean(axis=0).astype('int')
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    M = cv2.getRotationMatrix2D((np.mean(points[0, :]), np.mean(points[1,])), rect[2], 1)
    M_inv = cv2.getRotationMatrix2D((np.mean(points[0, :]), np.mean(points[1,])), -rect[2], 1)
    rot_rect = M.dot(points_ones.T).T.astype('int')
    rot_img = cv2.warpAffine(img,M,(img.shape[0],img.shape[1]))
    rot_img = rot_img if len(rot_img.shape) == 3 else np.expand_dims(rot_img,axis=2) #unsqueeze bw
    minx,maxx = np.min(rot_rect[:,0]),np.max(rot_rect[:,0])
    miny,maxy = np.min(rot_rect[:,1]),np.max(rot_rect[:,1])
    rot_crop = rot_img[miny-pads[0]:maxy+pads[0],minx - pads[0]: maxx + pads[0],:]
    zeros = np.zeros(img.shape,dtype='uint8')
    zeros[miny-pads[0]:maxy+pads[0],minx - pads[0]: maxx + pads[0],:]= rot_crop
    zeros = cv2.warpAffine(zeros,M_inv,dsize=img.shape[:-1])
    crop_wings = crop_dim[0]//2,crop_dim[1]//2
    zeros = zeros[max(cY-crop_wings[0],0) : min(cY+crop_wings[0],img.shape[0]),max(cX-crop_wings[1],0) : min(cX+crop_wings[1],img.shape[1])]
    if not zeros.shape == crop_dim:
        zeros = cv2.resize(zeros,crop_dim)
    return zeros

def get_file_pairs_2021(data_dir,split):
    file_pairs = get_file_pairs(data_dir,split)
    regex_string = r'202[0-9]-'
    year_finder = re.compile(regex_string)
    data_pairs_2021 = {}
    for front, files in file_pairs.items():
        ma = year_finder.search(front)
        if ma:
            if ma.group() == '2020-':
                pass
            elif ma.group() == '2021-':
                data_pairs_2021[front] = files
            else:
                print("warning: unknown matched year")
                print(front, ma.group())

        else:
            print("warning: unknown year")
            print(front)
    return data_pairs_2021

data_dir = '/pers_files/Combined_final/Filet'

base_dir = "/pers_files/Combined_final/Filet/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_4_output"
coco_dicts = data_utils.get_data_dicts(data_dir, 'train', file_pairs=get_file_pairs_2021(data_dir, 'train'))

data_utils.register_data('filet', ['train'], coco_dicts, {'thing_classes' : ['filet']})



cfg_fp = base_dir + "/cfg.yaml"
chk_dir = base_dir + "/best_model.pth"
#fig, axs = plt.subplots(2, 2)
ls_to_df = []
with torch.cuda.device(1):
    predictor = Filet_ModelTester(cfg_fp,chk_dir)
cols = ['file','ioubig1','ioubig2','n_inst_gt','n_inst_pred']
file_pairs = get_file_pairs_2021(data_dir,'train')

for front,files in file_pairs.items():
    if files[0].endswith(".jpg"):
        jpg, json_file = files
    else:
        json_file, jpg = files
    img = os.path.join(data_dir, 'train', jpg)
    input_img = cv2.imread(img)
    pred_output = predictor(input_img)
    n_inst_pred = len(pred_output['instances'])
    ious = np.zeros(n_inst_pred)
    with open(os.path.join(data_dir,'train',json_file)) as fp:
        gt_dict = json.load(fp)
    n_inst_gt = len(gt_dict['shapes'])
    masks = np.zeros((n_inst_gt,1024,1024))
    for i in range(n_inst_gt):
        masks[i] = polygon2mask((1024,1024),np.flip(np.array(gt_dict['shapes'][i]['points']),axis=1))
    masks_pred = pred_output['instances'].pred_masks.to('cpu').numpy().astype('uint8')
    is_big_pred = masks_pred.sum(axis=2).sum(axis=1)>22000
    ious = get_ious(masks,masks_pred[is_big_pred])
    ords = np.argsort(ious)[::-1]
    if len(ious):
        best = ious[ords[0]]
        if len(ious) == 1:
            runup = best
        else:
            runup = ious[ords[1]]
        df_row_dict = { col : val for col,val in zip(cols,[front,best,runup,n_inst_gt,n_inst_pred])}
        ls_to_df.append(df_row_dict)
        for i in range(n_inst_pred):
            _, contours, hierarchy = cv2.findContours(masks_pred[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt)>22000:
                iou = find_best_iou(masks,masks_pred[i])
                crop_img = crop_out_seg(input_img,cnt,pads = (7,7))
                mask_white = np.expand_dims(masks_pred[i]*255,axis=2)
                crop_mask = crop_out_seg(mask_white,cnt,pads = (7,7))
                cv2.imwrite(f"/pers_files/mask_data/{front}_ins{i}.jpg",crop_img)
                cv2.imwrite(f"/pers_files/mask_data/{front}_ins{i}mask.jpg",crop_mask)
                json_new = deepcopy(gt_dict)
                json_new['shapes'] = [{}]
                json_new['shapes'][0]['shape_type'] = "value"
                json_new['shapes'][0]['points'] = []
                json_new['shapes'][0]['label'] = iou
                with open(os.path.join(f"/pers_files/mask_data/{front}_ins{i}.json"),'w+') as fp:
                    json.dump(json_new,fp)
df = pd.DataFrame(ls_to_df)
df.to_csv(f"/pers_files/mask_data/iou_statistics")
print(df.mean())

