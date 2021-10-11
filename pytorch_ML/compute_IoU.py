import torch
from detectron2.structures import pairwise_iou
from torchvision.transforms import ColorJitter,RandomAffine,Compose
from numba import prange,njit
import numpy as np


class MaskScoreComputer():
    def __init__(self,score_fun):
        scores_implemented = ['d2_near','iou']
        assert score_fun in scores_implemented
        self.score_fun = score_fun

    @njit(parallel=True)
    def get_score(self,masks_gt, masks_pred):
        n_inst_pred = len(masks_pred)
        res = np.zeros(n_inst_pred)
        for i in prange(n_inst_pred):
            res[i] = self.find_best_score(masks_gt, masks_pred[i])
        return res

    @njit
    def find_best_score(self, gt_masks, pred_mask):
        n_inst = len(gt_masks)
        score_masks = np.zeros(n_inst)
        for i in range(n_inst):
            score_masks[i] = self.score(pred_mask,gt_masks[i])
        return np.max(score_masks)

    def score(self,out,target):
        raise NotImplementedError

class MaskIOUComputer(MaskScoreComputer):
    @njit
    def score(self, out,target):
        inter = np.logical_and(target, out).sum()
        union = np.logical_or(target, out).sum()
        return inter / union

class MaskD2NearComputer(MaskScoreComputer):
    @njit
    def d2_wrong_right(self,right,wrong):
        coords_r = np.array(right.nonzero()).transpose()
        coords_w = np.array(wrong.nonzero()).transpose()
        result = np.zeros(coords_w.shape[0])
        for i in range(coords_w.shape[0]):
            dists = np.sqrt(np.sum((coords_w[i] - coords_r)**2,axis=1))
            result[i] = np.min(dists)
        return result

    @njit
    def score(self,out,target):
        '''
        computes the average sq dist from predicted pixel, to nearest correct pixel with same 0/1 label.
        Value is negative to make it a score.
        '''
        tp = target * out
        tn = (1 - target) * (1 - out)
        fp = (1 - target) * out
        fn = target * (1 - out)
        arr_n = d2_wrong_right(tn,fn)
        arr_p = d2_wrong_right(tp,fp)
        denom = target.shape[0] * target.shape[1]
        result = (arr_n.sum() + arr_p.sum()) / denom
        return  - result





@njit(parallel = True)
def get_ious(masks_gt,masks_pred) -> np.ndarray:
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


def d2_near(out : np.ndarray ,target : np.ndarray ):
    tp = target * out
    tn = (1 - target) * (1 - out)
    fp = (1 - target) * out
    fn = target * (1 - out)
    arr_n = d2_wrong_right(tn,fn)
    arr_p = d2_wrong_right(tp,fp)
    denom = target.shape[0] *target.shape[1] * np.sqrt((target.shape[0]**2 + target.shape[1]**2))
    result = (arr_n.sum() + arr_p.sum()) / denom
    return result

#@njit
def d2_wrong_right(right,wrong):
    coords_r = np.array(right.nonzero()).transpose()
    coords_w = np.array(wrong.nonzero()).transpose()
    result = np.zeros(coords_w.shape[0])
    for i in range(coords_w.shape[0]):
        dists = np.sqrt(np.sum((coords_w[i] - coords_r)**2,axis=1))
        result[i] = np.min(dists)
    return result

class IOUComputerTorch():
    def get_score(self,masks_gt, masks_pred):
        n_inst_pred = len(masks_pred)
        res = torch.zeros(n_inst_pred,device = masks_pred.device)
        for i in range(n_inst_pred):
            res[i] = self.find_best_score(base_mask = masks_gt, mask_batch = masks_pred[i] )
        return res

    def get_score_by_gts(self,masks_gt, masks_pred):
        n_inst_gt = len(masks_gt)
        res = torch.zeros(n_inst_gt,device = masks_pred.device)
        for i in range(n_inst_gt):
            res[i] = self.find_best_score(base_mask=masks_gt[i], mask_batch = masks_pred)
        return res

    def find_best_score(self, base_mask,mask_batch):
        base_mask_unsq = base_mask.unsqueeze(0)
        inters = torch.logical_and(base_mask_unsq,mask_batch).sum(axis=(1,2))
        unions =torch.logical_or(base_mask_unsq,mask_batch).sum(axis=(1,2))
        scores = (inters / unions).max()
        return scores




#        if augs is None:
#            self.augs =  Compose([ColorJitter(brightness=1,contrast=1,saturation=1),RandomAffine(degrees=(-10,10))])
#        else:
#            self.augs = augs

class Inconsistency_Mask_Score():
    def __init__(self,min_size = 0,top_iou_nr = None,square_score = False):
        self.iou_computer = IOUComputerTorch()
        self.min_size = min_size
        self.top_iou_nr = top_iou_nr
        self.square_score = square_score
    def __call__(self,prime_masks : torch.Tensor,aug_masks_ls : list) -> torch.Tensor:
        are_large_prime_masks = prime_masks.sum(axis=(1,2)) > self.min_size
        large_prime_masks = prime_masks[are_large_prime_masks]
        results = torch.zeros((len(aug_masks_ls),len(large_prime_masks)))
        for i,aug_masks in enumerate(aug_masks_ls):
            results[i] = self.iou_computer.get_score_by_gts(masks_gt=large_prime_masks,masks_pred=aug_masks_ls[i])
        if self.square_score:
            loss = 1-results
            sq_loss = (loss**2).mean(axis=0)
            avg_iou = 1-torch.sqrt(sq_loss)
        else:
            avg_iou = results.mean(axis=0)

        if self.top_iou_nr is not None:
            if are_large_prime_masks.sum()>self.top_iou_nr:
                avg_iou = avg_iou.sort(descending=True)[0][:self.top_iou_nr]
        mean_avg_iou = avg_iou.mean()
        return mean_avg_iou
