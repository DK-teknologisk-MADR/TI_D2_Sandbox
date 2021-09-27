import os
import time
from copy import deepcopy
from detectron2.data import build_detection_test_loader
from detectron2_ML.trainers import TI_Trainer, Hyper_Trainer
from detectron2.evaluation import inference_on_dataset , DatasetEvaluator
import detectron2_ML.hooks as hooks
from detectron2.data.detection_utils import Instances
from detectron2_ML.hooks import StopAtIterHook
from pytorch_ML.compute_IoU import get_ious
from detectron2_ML.pruners import SHA
import pandas as pd
import time
from math import floor
import numpy as np
# install dependencies:
import datetime
from pycocotools.mask import decode
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
        for i, xput_dicts in enumerate( zip(inputs,outputs) ) :
            input_dict, output_dict = xput_dicts
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







