import numpy as np
import scipy

def get_r_mean_std(iou_arr):
    r_vals = np.where(iou_arr>0,iou_arr,0.01)
    r_vals = scipy.special.logit(r_vals)
    means , stds = r_vals.mean() , r_vals.std()
    return means, stds


def normalize_ious(iou_arr):
    r_vals = np.where(iou_arr>0,iou_arr,0.01)
    r_vals = scipy.special.logit(r_vals)
    normed_r = (r_vals -r_mean) / r_std
    iou_normed = scipy.special.expit(normed_r)
    return iou_normed


def rectify_ious(iou_arr):
    iou_normed = np.where(iou_arr>0.95,1,iou_arr)
    iou_normed = np.where(iou_normed<0.90,0,iou_normed)
    iou_normed = np.where(np.logical_and(iou_normed>0.90, iou_normed < 0.95) ,iou_normed * 20 - 18,iou_normed)
    return iou_normed
