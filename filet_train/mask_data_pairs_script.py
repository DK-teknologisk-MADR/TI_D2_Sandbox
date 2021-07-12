import os
import re
from filet_train.pytorch_ML.compute_IoU import find_best_iou, get_ious
import json
from copy import deepcopy
import torch
from torch.nn import Conv2d
from detectron2_ML import data_utils
import cv2
from detectron2_ML.data_utils import get_file_pairs
import pandas as pd
import matplotlib
from skimage.draw import polygon2mask
import numpy as np
matplotlib.use('TkAgg')
torch.cuda.device(0)
from filet_train.Filet_kpt_Predictor import Filet_ModelTester
device = 'cuda:0'
data_dir = '/pers_files/Combined_final/Filet'
base_dir = "/pers_files/Combined_final/Filet/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_4_output"
output_dir = "/pers_files/mask_pad_data19_norm"
os.mkdir(output_dir)
class Cropper():
    def __init__(self,crop,pad,device = 'cuda:0'):
        self.crop = crop
        self.conv = Conv2d(1, 1, (pad, pad), bias=False,padding=(pad//2,pad//2)).requires_grad_(False) #to be put in predictor class
        self.conv = self.conv.to('cuda:0')
        self.conv.weight[0] = torch.ones_like(self.conv.weight[0],device=device)

    def crop_out_seg (self,img_ts,mask):
        mask_padded = self.conv(mask.unsqueeze(0).float()).squeeze(0)
        mask_padded = mask_padded.bool().long()
        crop_img = (img_ts * mask_padded)[:,self.crop[0][0]:self.crop[0][1],self.crop[1][0]:self.crop[1][1]]
        return crop_img

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
split = 'train'

crop_dim = [[200,1024-200],[0,1024]]

pixel_pad = 19
cr = Cropper(crop_dim,pixel_pad)

coco_dicts = data_utils.get_data_dicts(data_dir, 'train', file_pairs=get_file_pairs_2021(data_dir, split))
data_utils.register_data('filet', ['train'], coco_dicts, {'thing_classes' : ['filet']})
cfg_fp = base_dir + "/cfg.yaml"
chk_dir = base_dir + "/best_model.pth"

#fig, axs = plt.subplots(2, 2)
ls_to_df = []
with torch.cuda.device(0):
    predictor = Filet_ModelTester(cfg_fp,chk_dir)
cols = ['file','ioubig1','ioubig2','n_inst_gt','n_inst_pred']
i = 0
file_pairs = get_file_pairs_2021(data_dir,split)
milestones = np.arange(0.1,1,0.1)
milestones = milestones * len(file_pairs)
milestones = [int(i) for i in milestones]
print("starting to write files")
for front,files in file_pairs.items():
    if i in milestones:
        print(i)
    if files[0].endswith(".jpg"):
        jpg, json_file = files
    else:
        json_file, jpg = files
    img = os.path.join(data_dir, split, jpg)
    input_img = cv2.imread(img)
    pred_output = predictor(input_img)
    n_inst_pred = len(pred_output['instances'])
    ious = np.zeros(n_inst_pred)
    with open(os.path.join(data_dir,split,json_file)) as fp:
        gt_dict = json.load(fp)
    n_inst_gt = len(gt_dict['shapes'])
    masks = np.zeros((n_inst_gt,1024,1024))
    for i in range(n_inst_gt):
        masks[i] = polygon2mask((1024,1024),np.flip(np.array(gt_dict['shapes'][i]['points']),axis=1))
    masks_pred_ts = pred_output['instances'].pred_masks
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
            if is_big_pred[i]:
                iou = find_best_iou(masks,masks_pred[i])
                crop_img = cr.crop_out_seg(torch.tensor(input_img,device=device).permute(2,0,1),masks_pred_ts[i].unsqueeze(0)).permute(1,2,0)
                crop_img = crop_img.to('cpu').numpy().astype('float32')
                ch = masks_pred[i].mean(axis = 0)
                cw = masks_pred[i].mean(axis = 1)
                center = np.array(np.nonzero(masks_pred)).mean()
                #TODO:Centralize picture
                mask_white = np.expand_dims(masks_pred[i]*255,axis=2).astype('uint8')
                crop_mask = mask_white[crop_dim[0][0]:crop_dim[0][1],crop_dim[1][0]:crop_dim[1][1]]
                crop_img = cv2.resize(crop_img,(crop_dim[1][1]*3//4,crop_dim[0][1]*3//4))
                crop_mask = cv2.resize(crop_mask,(crop_dim[1][1]*3//4,crop_dim[0][1]*3//4))
                cv2.imwrite(f"{output_dir}/{front}_ins{i}.jpg",crop_img)
                cv2.imwrite(f"{output_dir}/{front}_ins{i}mask.jpg",crop_mask)
                json_new = deepcopy(gt_dict)
                json_new['shapes'] = [{}]
                json_new['shapes'][0]['shape_type'] = "value"
                json_new['shapes'][0]['points'] = []
                json_new['shapes'][0]['label'] = iou
                with open(os.path.join(f"{output_dir}/{front}_ins{i}.json"),'w+') as fp:
                    json.dump(json_new,fp)
    i += 1
df = pd.DataFrame(ls_to_df)
df.to_csv(f"/pers_files/mask_data/iou_statistics")