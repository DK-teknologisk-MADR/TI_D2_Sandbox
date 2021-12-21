import os
import re
from pytorch_ML.compute_IoU import  find_best_iou, get_ious
import json
from copy import deepcopy
import torch
from torch.nn import Conv2d
from detectron2_ML import data_utils
import cv2
from time import time
from torchvision.transforms import InterpolationMode
from torchvision.transforms import ToTensor,Normalize,Compose
from torchvision.transforms.functional import affine,resized_crop
from detectron2_ML.data_utils import get_file_pairs
import pandas as pd
import matplotlib
from skimage.draw import polygon2mask
import numpy as np
matplotlib.use('TkAgg')
torch.cuda.device(0)
from filet.mask_discriminator.transformations import centralize_img_wo_crop
from detectron2_ML.predictors import ModelTester #from filet.Filet_kpt_Predictor import Filet_ModelTester
import numba
device = 'cuda:0'
data_dir = '/pers_files/Combined_final/Filet'
base_dir = "/pers_files/Combined_final/Filet/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_4_output"
output_dir = "/pers_files/mask_data_raw_good_score"

pic_dim = 1024
class Cropper():
    def __init__(self,crop,pad,device = 'cuda:0'):
        self.crop = crop
        self.conv = Conv2d(1, 1, (pad, pad), bias=False,padding=(pad//2,pad//2)).requires_grad_(False) #to be put in predictor class
        self.conv = self.conv.to(device)
        self.conv.weight[0] = torch.ones_like(self.conv.weight[0],device=device)

    def crop_out_seg (self,img_ts,mask):
        mask_padded = self.conv(mask.unsqueeze(0).float()).squeeze(0)
        mask_padded = mask_padded.bool().long()
        crop_img = (img_ts * mask_padded)[:,self.crop[0][0]:self.crop[0][1],self.crop[1][0]:self.crop[1][1]]
        return crop_img

class PreProcessor():
    def __init__(self,crop,pad,mean,std):
        self.to_ts = ToTensor()
        self.norm = Normalize(mean=mean,std=std)
        self.torchvision_transforms = Compose([self.to_ts,self.norm])
        self.crop = crop
        self.conv = Conv2d(1, 1, (pad, pad), bias=False,padding=(pad//2,pad//2),).requires_grad_(False) #to be put in predictor class
        self.conv.weight[0] = torch.ones_like(self.conv.weight[0],device='cpu')

    def preprocess(self,img,raw_mask):
        '''
        BGR_IMG
        '''
        mask = np.asarray(raw_mask>0,dtype='uint8')
        mask, centroids, area, discard_flag = self.get_component_analysis(mask)
        height, width, ch = img.shape
        t_1 = width // 2 - centroids[0]
        t_2 = height // 2 - centroids[1]
        T = np.float32([[1, 0, t_1], [0, 1, t_2]])
        mask_center = cv2.warpAffine(mask, T, (width, height))[self.crop[0][0]: self.crop[0][1], self.crop[1][0]: self.crop[1][1]]
        img_center = cv2.warpAffine(img, T, (width, height))[self.crop[0][0]: self.crop[0][1], self.crop[1][0]: self.crop[1][1]]
        img_ts = self.torchvision_transforms(img_center)
        mask_center = torch.tensor(mask_center,dtype=torch.float).unsqueeze(0)

        mask_padded = self.conv(mask_center.unsqueeze(0)).squeeze(0)
        mask_padded = mask_padded.bool().long()
        img_masked = (img_ts * mask_padded)
        full_output = torch.vstack([img_masked,mask_center])
        return full_output


    def get_component_analysis(self, mask):
        '''
        returns:mask of biggest component
        centroids in format (x from left, y from top)
        largest area in pixel count
        discard_flag hints whether its too weird to keep.
        '''
        discard_flag = False
        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
        # centroids in x from left, y from top.
        largest_components = stats[1:, 4].argsort()[::-1] + 1
        largest_area = stats[largest_components[0], 4]
        proportions = stats[1:, 4] / largest_area
        if np.any(np.logical_and(proportions > 0.2, proportions < 1)):
            discard_flag = True
        mask = np.where(labels == largest_components[0], 1, 0).astype('uint8')
        centroids = centroids[largest_components[0]]
        return mask, centroids, largest_area, discard_flag

    # test_mask = '/pers_files/mask_pad_data19/test/robotcell_all1_color_2021-02-04-09-28-12_ins4mask.jpg'
    # test_img = '/pers_files/mask_pad_data19/test/robotcell_all1_color_2021-02-04-09-28-12_ins4.jpg'
    # mask = cv2.imread(test_mask,cv2.IMREAD_GRAYSCALE)
    # y  = cv2.imread(test_img)
    # jitter = np.random.randint(0,255,y.shape).astype('uint8')
    # y = np.where(y>0,y,jitter)
    # cv2.imshow("win",y)
    # cv2.waitKey()
    # prp = PreProcessor(None,19,[0,0,0],[1,1,1])
    #
    # out = prp.preprocess(y,mask)
    # out.shape
    # out.to('cpu').permute(1,2,0).numpy()[:,:,:3].shape
    # cv2.imshow("win",out.to('cpu').permute(1,2,0).numpy()[:,:,:3])
    # cv2.waitKey()


def generate_masks_and_json(split):
    pic_dim = 1024
    data_dir = '/pers_files/Combined_final/Filet'
    base_dir = "/pers_files/Combined_final/Filet/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_4_output"
    output_dir = f"/pers_files/mask_data_raw_TV_good_score/{split}"
    os.makedirs(output_dir,exist_ok=False)
    coco_dicts = data_utils.get_data_dicts(data_dir, split, file_pairs=get_file_pairs_2021(data_dir, split))
    data_utils.register_data('filet', [split], coco_dicts, {'thing_classes': ['filet']})
    cfg_fp = base_dir + "/cfg.yaml"
    chk_dir = base_dir + "/best_model.pth"

    # fig, axs = plt.subplots(2, 2)
    ls_to_df = []
    with torch.cuda.device(0):
        predictor = ModelTester(cfg_fp, chk_dir)
    cols = ['file', 'ioubig1', 'ioubig2', 'n_inst_gt', 'n_inst_pred']
    i = 0
    file_pairs = get_file_pairs_2021(data_dir, split)
    milestones = np.arange(0.1, 1, 0.1)
    milestones = milestones * len(file_pairs)
    milestones = [int(i) for i in milestones]

    cent_img_frames = []  # just for testing
    iou_dict = {}
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
        with open(os.path.join(data_dir,split,json_file)) as fp:
            gt_dict = json.load(fp)
        n_inst_gt = len(gt_dict['shapes'])
        masks = np.zeros((n_inst_gt,pic_dim,pic_dim))
        for i in range(n_inst_gt):
            masks[i] = polygon2mask((pic_dim,pic_dim),np.flip(np.array(gt_dict['shapes'][i]['points']),axis=1))
        has_good_score = pred_output['instances'].scores > 0.8
        masks_pred =pred_output['instances'][has_good_score].pred_masks.to('cpu').numpy().astype('uint8')
        is_big_pred = masks_pred.sum(axis=2).sum(axis=1)>22000
        ious = get_ious(masks,masks_pred[is_big_pred])
        ords = np.argsort(ious)[::-1]
        if len(ious):
            best = ious[ords[0]]
            if len(ious) == 1:
                runup = best
            else:
                runup = ious[ords[1]]
            df_row_dict = { col : val for col,val in zip(cols,[front,best,runup,n_inst_gt,len(masks_pred)])}
            ls_to_df.append(df_row_dict)
            for i in range(len(masks_pred)):
                if is_big_pred[i]:
                    iou = find_best_iou(masks, masks_pred[i])
                    iou_dict[f"{output_dir}/{front}_ins{i}"] = iou
#                    cv2.imwrite(f"{output_dir}/{front}_ins{i}.jpg", crop_img)
                    cv2.imwrite(f"{output_dir}/{front}_ins{i}mask.jpg", 255 * masks_pred[i])
                    json_new = deepcopy(gt_dict)
                    json_new['shapes'] = [{}]
                    json_new['shapes'][0]['shape_type'] = "value"
                    json_new['shapes'][0]['points'] = []
                    json_new['shapes'][0]['label'] = iou
                    with open(os.path.join(f"{output_dir}/{front}_ins{i}mask.json"), 'w+') as fp:
                        json.dump(json_new, fp)



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

generate_masks_and_json('val')
generate_masks_and_json('train')
#
#
#
#
# crop_dim = [[0,pic_dim],[0,pic_dim]]#crop_dim = [[200,pic_dim-200],[0,pic_dim]]
#
# pixel_pad = 19
# cr = Cropper(crop_dim,pixel_pad)
#
# coco_dicts = data_and_file_utils.get_data_dicts(data_dir, 'train', file_pairs=get_file_pairs_2021(data_dir, split))
# data_and_file_utils.register_data('filet', ['train'], coco_dicts, {'thing_classes' : ['filet']})
# cfg_fp = base_dir + "/cfg.yaml"
# chk_dir = base_dir + "/best_model.pth"
#
# #fig, axs = plt.subplots(2, 2)
# ls_to_df = []
# with torch.cuda.device(0):
#     predictor = ModelTester(cfg_fp,chk_dir)
# cols = ['file','ioubig1','ioubig2','n_inst_gt','n_inst_pred']
# i = 0
# file_pairs = get_file_pairs_2021(data_dir,split)
# milestones = np.arange(0.1,1,0.1)
# milestones = milestones * len(file_pairs)
# milestones = [int(i) for i in milestones]
#
# cent_img_frames= []#just for testing
# iou_dict = {}
# print("starting to write files")
# for front,files in file_pairs.items():
#     if i in milestones:
#         print(i)
#     if files[0].endswith(".jpg"):
#         jpg, json_file = files
#     else:
#         json_file, jpg = files
#     img = os.path.join(data_dir, split, jpg)
#     input_img = cv2.imread(img)
#     pred_output = predictor(input_img)
#     n_inst_pred = len(pred_output['instances'])
#     ious = np.zeros(n_inst_pred)
#     with open(os.path.join(data_dir,split,json_file)) as fp:
#         gt_dict = json.load(fp)
#     n_inst_gt = len(gt_dict['shapes'])
#     masks = np.zeros((n_inst_gt,pic_dim,pic_dim))
#     for i in range(n_inst_gt):
#         masks[i] = polygon2mask((pic_dim,pic_dim),np.flip(np.array(gt_dict['shapes'][i]['points']),axis=1))
#     masks_pred_ts = pred_output['instances'].pred_masks
#     masks_pred = pred_output['instances'].pred_masks.to('cpu').numpy().astype('uint8')
#     is_big_pred = masks_pred.sum(axis=2).sum(axis=1)>22000
#     ious = get_ious(masks,masks_pred[is_big_pred])
#     ords = np.argsort(ious)[::-1]
#     if len(ious):
#         best = ious[ords[0]]
#         if len(ious) == 1:
#             runup = best
#         else:
#             runup = ious[ords[1]]
#         df_row_dict = { col : val for col,val in zip(cols,[front,best,runup,n_inst_gt,n_inst_pred])}
#         ls_to_df.append(df_row_dict)
#         for i in range(n_inst_pred):
#             if is_big_pred[i]:
#                 iou = find_best_iou(masks,masks_pred[i])
#                 iou_dict[f"{output_dir}/{front}_ins{i}"] = iou
#                 crop_img = cr.crop_out_seg(torch.tensor(input_img,device=device).permute(2,0,1),masks_pred_ts[i].unsqueeze(0)).permute(1,2,0)
#                 crop_img = crop_img.to('cpu').numpy().astype('float32')
#                 crop_img ,translations , flag = centralize_img_wo_crop(crop_img)
#                 #TODO:Centralize picture
#                 mask_white = np.expand_dims(masks_pred[i]*255,axis=2).astype('uint8')
#                 mask_white, _  , flag= centralize_img_wo_crop(mask_white,translations)
#                 if flag:
#                     print(f"flag raised for {output_dir}/{front}_ins{i}.jpg")
#                 non_zero_coords = np.array(np.nonzero((crop_img)))
#                 h_min, w_min = non_zero_coords.min(axis=1)[:2]
#                 h_max, w_max = non_zero_coords.max(axis=1)[:2]
#                 print([w_min,w_max,h_min,h_max])
#                 cent_img_frames.append([w_min,w_max,h_min,h_max])
#                 crop_dim_end = [[200, pic_dim - 200], [100, pic_dim-100]]
#                 crop_mask = mask_white[crop_dim_end[0][0]:crop_dim_end[0][1],crop_dim_end[1][0]:crop_dim_end[1][1]]
#                 crop_img = crop_img[crop_dim_end[0][0]:crop_dim_end[0][1],crop_dim_end[1][0]:crop_dim_end[1][1]]
#                 crop_img = cv2.resize(crop_img,(crop_dim_end[1][1]*3//4,crop_dim_end[0][1]*3//4))
#                 crop_mask = cv2.resize(crop_mask,(crop_dim_end[1][1]*3//4,crop_dim_end[0][1]*3//4))
#                 cv2.imwrite(f"{output_dir}/{front}_ins{i}.jpg",crop_img)
#                 cv2.imwrite(f"{output_dir}/{front}_ins{i}mask.jpg",crop_mask)
#                 json_new = deepcopy(gt_dict)
#                 json_new['shapes'] = [{}]
#                 json_new['shapes'][0]['shape_type'] = "value"
#                 json_new['shapes'][0]['points'] = []
#                 json_new['shapes'][0]['label'] = iou
#                 with open(os.path.join(f"{output_dir}/{front}_ins{i}.json"),'w+') as fp:
#                     json.dump(json_new,fp)
# with open(os.path.join(f"{output_dir}/IOU_register.json"),'w+') as fp:
#     json.dump(iou_dict,fp)
# cent_img_frames = np.array(cent_img_frames)
# print("EXTREME VALUES")
# print(cent_img_frames.min(axis=0))
# print(cent_img_frames.max(axis=0))
#
# df = pd.DataFrame(ls_to_df)
# df.to_csv(f"/pers_files/mask_data/iou_statistics")
#
#
#
# x = np.random.randint(0,2,(756,756))
# x = np.zeros((756,756),dtype='uint8')
# height,width = x.shape
# x[256:300,256:300] = 1
# x[244:300,0:55] = 1
# x[0:300,0:400] = 1
#
# x[744:756,110:756] = 1
#
# mask = x
# cv2.imshow("window",x*255)
# cv2.waitKey()
#
# connectivity = 4
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
# # centroids in x from left, y from top.
# largest_components = stats[1:, 4].argsort()[::-1] + 1
# largest_area = stats[largest_components[0], 4]
# proportions = stats[1:, 4] / largest_area
# if np.any(np.logical_and(proportions > 0.2, proportions < 1)):
#     discard_flag = True
# mask = np.where(labels == largest_components[0], 1, 0).astype('uint8')
# centroids = centroids[largest_components[0]]
#
#
# # You need to choose 4 or 8 for connectivity type
# # Perform the operation
