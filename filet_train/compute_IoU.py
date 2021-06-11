import torch
from detectron2.structures import pairwise_iou
from numba import prange,njit
import numpy as np

@njit(parallel = True)
def get_ious(masks_gt,masks_pred):
    n_inst_pred = len(masks_pred)
    res = np.zeros(n_inst_pred)
    for i in prange(n_inst_pred):
        res[i] = find_best_iou(masks_gt,masks_pred[i])
    return res
@njit
def find_best_iou(gt_masks,pred_mask):
    n_inst = len(gt_masks)
    iou_masks = np.zeros(n_inst)
    for i in range(n_inst):
        iou_masks[i] = iou_mask(gt_masks[i],pred_mask)
    return np.max(iou_masks)


@njit
def iou_mask(arr1,arr2):
    inter = np.logical_and(arr1,arr2).sum()
    union = np.logical_or(arr1,arr2).sum()
    return inter / union

