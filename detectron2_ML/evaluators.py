import os
import time
from copy import deepcopy
from itertools import zip_longest

import numpy
import torch
from detectron2.data import build_detection_test_loader
from detectron2_ML.trainers import TI_Trainer, Hyper_Trainer
from detectron2.evaluation import inference_on_dataset , DatasetEvaluator
import detectron2_ML.hooks as hooks
from detectron2.engine import DefaultPredictor
from detectron2.data.detection_utils import Instances
from torchvision.transforms import ColorJitter,RandomAffine,Normalize,ToTensor
from torchvision.transforms.functional import adjust_contrast,adjust_brightness,affine,_get_inverse_affine_matrix , adjust_saturation
from numpy.random import choice, randint,uniform
from cv2_utils.colors import RGB_TO_COLOR_DICT
from cv2_utils.cv2_utils import *
from detectron2_ML.trainers import TI_Trainer,Trainer_With_Early_Stop
from detectron2_ML.data_utils import get_data_dicts, register_data , get_file_pairs,sort_by_prefix
from spoleben_train.data_utils import get_data_dicts_masks
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2_ML.hooks import StopAtIterHook
from pytorch_ML.compute_IoU import get_ious
from detectron2_ML.pruners import SHA
import pandas as pd
from pytorch_ML.compute_IoU import Inconsistency_Mask_Score
from time import time
from math import floor
import numpy as np
# install dependencies:
import datetime
from pycocotools.mask import decode



#FOR TESTING#
cfg = get_cfg()

cfg.merge_from_file('/pers_files/spoleben/spoleben_09_2021/output_test/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_6_output/cfg.yaml')
cfg.OUTPUT_DIR = '/pers_files/spoleben/spoleben_09_2021/output_test/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_6_output'
cfg.MODEL.WEIGHTS = '/pers_files/spoleben/spoleben_09_2021/output_test/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_6_output/best_model.pth'
cfg.INPUT.MIN_SIZE_TEST=450
splits = ['']
data_dir = '/pers_files/spoleben/spoleben_09_2021/spoleben_batched' #"/pers_files/spoleben/FRPA_annotering/annotations_crop(180,330,820,1450)"
file_pairs = { split : sort_by_prefix(os.path.join(data_dir,split)) for split in splits }
#file_pairs = { split : get_file_pairs(data_dir,split,sorted=True) for split in splits }
COCO_dicts = {split: get_data_dicts_masks(data_dir,split,file_pairs[split]) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names = register_data('filet',splits,COCO_dicts,{'thing_classes' : ['spoleben']}) #register data by str name in D2 api
output_dir = f'/pers_files/spoleben/spoleben_09_2021/output_test2'
data = COCO_dicts[""]
#

class MeatPickEvaluator(DatasetEvaluator):

    def __init__(self,coco_dicts_ls,top_n_ious,min_size_gt = 0, min_size_pred = 0):
        self.top_n_ious = top_n_ious
        self.min_size_gt = min_size_gt
        self.min_size_pred = min_size_pred
        self.evaluations = []
        self.coco_dicts_ls = deepcopy(coco_dicts_ls)
        self.file_name_to_id = {}
        for i in range(len(coco_dicts_ls)):
            self.file_name_to_id[coco_dicts_ls[i]['file_name']] = i
    def reset(self):
        self.evaluations = []

    def process(self,inputs,outputs):
        total = len(inputs)
        start_time = time.time()
        print("INTPUTS ARE",inputs)
        for i, xput_dict_pair in enumerate( zip_longest(inputs,outputs,fillvalue=None) ) :
            input_dict, output_dict = xput_dict_pair

            pcts = floor((i-1 / total)*10) , floor((i / total)*10)
            if pcts[0] != pcts[1] and pcts[0]>0:
                print(f"COMPLETED {pcts[1]*10}% OF EVALUATION COMPUTATIONS. time from start:{time.time()-start_time}, time / instance: {(time.time()-start_time) / i}")
            print(input_dict,output_dict)
            fp = input_dict['file_name']
            ID = self.file_name_to_id[fp]
            gt_masks = [decode(annotation['segmentation']) for annotation in self.coco_dicts_ls[ID]['annotations']]
            gt_masks = np.stack(gt_masks,axis=0)
            output_instances = output_dict['instances']
            assert isinstance(output_instances,Instances)
            pred_masks = output_instances.pred_masks.to('cpu').numpy()
            print("gt_mask_shape",gt_masks.shape)
            print("pred_mask_shape",pred_masks.shape)

            are_large_gt_masks = gt_masks.sum(axis=(1,2)) > self.min_size_gt
            are_large_pred_masks = pred_masks.sum(axis=(1,2)) > self.min_size_pred
            print("large_gt_masks",are_large_gt_masks.sum())
            print("large_pred_masks",are_large_pred_masks.sum())
            nr_of_large_gt_masks = are_large_gt_masks.sum()
            nr_of_large_pred_masks = are_large_pred_masks.sum()
            gt_masks = gt_masks[are_large_gt_masks]
            pred_masks = pred_masks[are_large_pred_masks]
            sorted_ious = np.sort(get_ious(gt_masks,pred_masks))[::-1]
            nr_relevant_gt_masks = np.min([nr_of_large_gt_masks,self.top_n_ious])
            if len(sorted_ious)>self.top_n_ious:
                sorted_ious = sorted_ious[:nr_relevant_gt_masks]
            result = sorted_ious.sum() / np.min([nr_relevant_gt_masks,self.top_n_ious])
            self.evaluations.append(result)
        print(f"COMPLETED {100}% OF EVALUATION COMPUTATIONS. time from start:{time.time() - start_time}, time / instance: {(time.time() - start_time) / total}")
    def evaluate(self):
        # save self.count somewhere, or print it, or return it.
        print('EVALUATOR CALLED')
        return {'segm' : {'best_ious' : np.mean(self.evaluations)}}





class Consistency_Evaluator(DatasetEvaluator):
    def __init__(self,predictor_cfg, coco_dicts_ls, top_n_ious,img_size,device,pads_pct_hw = (.10,.10), min_size_gt = 0,min_size_pred = 0, min_size_incon = 0):
        '''
        Computes consistency loss of a dataset ( coco_dicts_ls), based on outputs, created from a DefaultPredictor from param predictor_cfg.
        Only base output masks larger than min_size_incon is considered. Also only masks within a frame defied by pads_pct_hw is considered
        '''
        #TODO:: Remove coco_dicts_ls from init and replace with cfg.dataset and get from register.
        self.top_n_ious = top_n_ious
        self.min_size_gt = min_size_gt
        self.min_size_pred = min_size_pred
        self.min_size_incon = min_size_incon
        self.pred = DefaultPredictor(predictor_cfg)
        self.h,self.w = img_size[0],img_size[1]
        self.evaluations = []
        self.aug_nr = 4 #HARDCODED DO NOT CHANGE ATM
        self.angles = [7, -7, 7, -7]
        self.translate = [(37, 41), (43, -49), (-45, 39), (-39, 45)]
        self.coco_dicts_ls = deepcopy(coco_dicts_ls)
        self.file_name_to_id = {}
        self.pads_pct_hw = pads_pct_hw
        self.pad_frame = torch.zeros(img_size,device='cpu',dtype=torch.bool)
        pad_x,pad_y = int(img_size[0] * pads_pct_hw[0] / 2) , int(pads_pct_hw[1] * img_size[1] / 2)
        self.to_norm = Normalize(mean=[0,0,0],std=[255,255,255])
        print(pad_x,pad_y,self.pad_frame.shape)
        self.pad_frame[:pad_x] = True
        self.pad_frame[-pad_x:] = True
        self.pad_frame[:,:pad_y] = True
        self.pad_frame[:,-pad_y:] = True
        self.device=device
        self.to_ts = ToTensor()
        self.loss = Inconsistency_Mask_Score(min_size=min_size_gt, top_iou_nr=top_n_ious)
        self.nr_of_pixels = img_size[0] * img_size[1]
        self.consistency_scores = []
        for i in range(len(coco_dicts_ls)):
            self.file_name_to_id[coco_dicts_ls[i]['file_name']] = i
        self.evaluation_results = {}
        self.evaluations = []

    def reset(self):
        self.evaluation_results = {}
        self.evaluations = []

    def process(self,inputs,outputs=[]):
        total = len(inputs)
        start_time = time()
        print("INTPUTS ARE",inputs)
        for i, xput_dicts in enumerate( zip_longest(inputs,outputs,fillvalue=None) ) :
            print("INSIDE")
            input_dict, output_dict = xput_dicts
            pcts = floor((i-1 / total)*10) , floor((i / total)*10)
            if pcts[0] != pcts[1] and pcts[0]>0:
                print(f"COMPLETED {pcts[1]*10}% OF EVALUATION COMPUTATIONS. time from start:{time()-start_time}, time / instance: {(time()-start_time) / i}")
            fp = input_dict['file_name']
            ID = self.file_name_to_id[fp]
            img = input_dict['image']
#            gt_masks = [decode(annotation['segmentation']) for annotation in self.coco_dicts_ls[ID]['annotations']]
#            gt_masks = np.stack(gt_masks,axis=0)
            img_ts = self.tr_ts(img)
            img_ts = img_ts.expand(size = (1 + self.aug_nr, 3, img.shape[1], img.shape[2])).clone()
            with torch.no_grad():
                start = time()
                img_ts.to('cuda:0')
                img_ts[1:3, ] = adjust_brightness(img_ts[1:3, ], brightness_factor=0.7)
                img_ts[4:, ] = adjust_brightness(img_ts[4:, ], brightness_factor=1.3)

                img_ts[1,] = adjust_contrast(img_ts[1,], contrast_factor=0.6)
                img_ts[4,] = adjust_contrast(img_ts[4,], contrast_factor=1.4)

                img_ts[2, ] = adjust_saturation(img_ts[2, ], saturation_factor=0.7)
                img_ts[4, ] = adjust_saturation(img_ts[4, ], saturation_factor=1.3)
                for i in range(self.aug_nr):
                    img_ts[1 + i] = affine(img_ts[1 + i], angle=self.angles[i], translate=self.translate[i], scale=1,
                                           shear=[0, 0])
                img_ts_ls = img_ts.split(1)
                img_ts_dict = [{'image': img.squeeze(0) * 255} for img in img_ts_ls]
                output = self.pred.model(img_ts_dict)
                pred_masks = [output_dict['instances'].pred_masks for output_dict in output]
                pred_masks = [pred_mask_batch[pred_mask_batch.sum(axis=(1,2)) > self.min_size_incon] for pred_mask_batch in pred_masks]
                pred_masks[0] = self.get_center_masks(pred_masks[0])

                for i in range(self.aug_nr):
                    pred_masks[1 + i] = affine(pred_masks[1 + i].unsqueeze(1), angle=0,
                                               translate=(- self.translate[i][0], - self.translate[i][1]), scale=1,
                                               shear=[0, 0]).squeeze_(1)
                for i in range(self.aug_nr):
                    pred_masks[1 + i] = affine(pred_masks[1 + i].unsqueeze(1), angle=-self.angles[i], translate=(0, 0),
                                               scale=1, shear=[0, 0]).squeeze_(1)
                torch.cuda.synchronize()
                end = time()
            print(end - start)
            #masks_cpu = [mask.to('cpu').numpy() for mask in pred_masks]
            #img = tensor_pic_to_imshow_np(img_ts[0])
            #over = put_mask_overlays(img, masks_cpu[4])
#
#             output_instances = output_dict['instances']
#             assert isinstance(output_instances,Instances)
#             pred_masks = output_instances.pred_masks
#             pred_masks = self.get_center_masks(pred_masks)
#             img = input_dict['image']
#             aug_ls =[]
# #           print("gt_mask_shape",gt_masks.shape)
#             print("pred_mask_shape",pred_masks.shape)
            print(self.loss(pred_masks[0], pred_masks[1:]))
            self.evaluation_results[fp] = float(self.loss(pred_masks[0],pred_masks[1:]).numpy())


    def get_center_masks(self,masks_ts):
        pad_frame_gpu = self.pad_frame.to(self.device).unsqueeze_(0).expand_as(masks_ts)
        amount_in_edge = torch.logical_and(masks_ts,pad_frame_gpu).sum(axis=(1,2))
        masks_area = masks_ts.sum(axis=(1,2))
        center_masks = masks_ts[amount_in_edge / masks_area < 0.1 ]
        return center_masks

    def evaluate(self):
        result_per_file = self.evaluation_results.copy()
        values = np.array(list(result_per_file.values()))
        overall_score = values.mean()
        # save self.count somewhere, or print it, or return it.
        result = {'file_score_dict' : result_per_file,
                  'dataset_score' : overall_score,
                  }

        return result

    def tr_ts(self,img):
        if isinstance(img,numpy.ndarray):
            return self.to_ts(img)
        elif isinstance(img,torch.Tensor):
            return self.to_norm(img.float())

#tensor_pic_to_imshow_np(data['image'])
