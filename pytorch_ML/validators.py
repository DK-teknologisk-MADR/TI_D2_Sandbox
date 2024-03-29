import torch
from sklearn.metrics import f1_score as f1_score_sk
import numpy as np






def prec_rec_spec_neqprec_scores(y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 2. 0 <= val <= 1
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)


    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    spec = tn / (tn + fp + epsilon)
    neg_prec = tn / (tn + fn + epsilon)

    result = torch.hstack((precision,recall,spec,neg_prec))
    result.requires_grad = False
    return result

def prec_rec_scores(y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 2. 0 <= val <= 1
    '''
    assert y_true.ndim <3
    if y_true.ndim == 2:
        if y_true.shape[1] == 1:
            y_true = y_true.flatten()
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)


    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
#    spec = tn / (tn + fp + epsilon)
#    neg_prec = tn / (tn + fn + epsilon)

    result = torch.hstack((precision,recall))
    result.requires_grad = False
    return result

def f1_score(y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
    prec_rec = prec_rec_scores(y_true,y_pred)
    epsilon = 1e-7
    f1 = 2* (prec_rec[0]*prec_rec[1]) / (prec_rec[0] + prec_rec[1] + epsilon)
    f1.requires_grad = False
    return f1


class F1_Score_Osc():
    def __init__(self,nr_of_classes : int ,th : float = 0.5):
        self.th = th
        self.nr_of_classes = nr_of_classes

    def __call__(self,y_pred:torch.Tensor, y_true:torch.Tensor):
        assert y_true.ndim < 3
        if y_true.ndim == 2:
            if y_true.shape[1] == 1:
                y_true = y_true.flatten()
            else:
                raise ValueError(f"y true has bad shape {y_true.shape[1]}")
        assert y_pred.ndim == 2

        y_pred = y_pred.argmax(dim=1) * torch.any(y_pred > self.th,dim=1) + torch.all(y_pred <= self.th,dim=1) * self.nr_of_classes
        y_pred_np = y_pred.to('cpu').numpy()
        y_true_np = y_true.to('cpu').numpy()
        print(np.array([y_true_np,y_pred_np,torch.all(y_pred <= self.th,dim=1)]))
        print("F1 scores are pr class are:" )
        print({i : f1 for i,f1 in enumerate(f1_score_sk(y_true=y_true_np,y_pred=y_pred_np,average=None))})
        return f1_score_sk(y_true=y_true_np,y_pred=y_pred_np,average='weighted')
#
#        F1s = torch.zeros(self.nr_of_classes + 1,dtype=torch.float32)
#        epsilon = 1e-7
#        for val_to_check in range(self.nr_of_classes + 1):
#            pred_is_val = (y_pred == val_to_check).long()
#            true_is_val = (y_true == val_to_check).long()
#            tp = (true_is_val * pred_is_val).sum().to(torch.float32)
#            tn = ((1 - true_is_val) * (1 - pred_is_val)).sum().to(torch.float32)
#            fp = ((1 - true_is_val) * pred_is_val).sum().to(torch.float32)
#            fn = (true_is_val * (1 - pred_is_val)).sum().to(torch.float32)
#            precision = tp / (tp + fp + epsilon)
#            recall = tp / (tp + fn + epsilon)
#            F1s[val_to_check] = 2 * (precision * recall) / (precision + recall + epsilon)



def f1_score_neg(y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
    y_true = torch.abs(1-y_true)

    y_pred = torch.abs(1-y_pred)
    result = f1_score(y_true,y_pred)
    return result


def worst_f1(y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
    assert y_true.ndim == 2
    assert y_pred.ndim == 2 or y_pred.ndim == 3
    result_len = y_true.shape[1]
    result= torch.zeros(result_len,requires_grad=False,device=y_true.device)
    for i in range(result_len):
        result[i] = f1_score(y_true[:,i],y_pred[:,i])

    min_result = torch.min(result)
    min_result.requires_grad = False
    return min_result


def mcc_score(y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
    #TODO: SHould assume y_pred is output from sigmoid
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    score = mcc_from_xps(tp,tn,fp,fn)
    return score

def mcc_from_xps(tp : torch.Tensor,tn : torch.Tensor,fp : torch.Tensor,fn : torch.Tensor):

    epsilon = 1e-7
    num = (tp*tn) - (fp*fn)
    den = (tp + fp)*(tp+fn) *(tn+fp) * (tn+fn)
    score = num / torch.sqrt(den)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    spec = tn / (tn + fp + epsilon)
    neg_prec = tn / (tn + fn + epsilon)
    print('prec: ',precision,"recall: ",recall,"spec: ",spec,"neg_prec:", neg_prec)
    return score


def tp_tn_fp_fn_from_th(outs,targets,th=0.5):
    '''
    Assumes y_pred is output from sigmoid, and output is batchwise,
    '''
    assert targets.ndim == 2
    assert outs.ndim == 2
    y_pred = (outs > th).to(torch.float32)
    tp = (targets * y_pred).sum().to(torch.float32)
    tn = ((1 - targets) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - targets) * outs).sum().to(torch.float32)
    fn = (targets * (1 - outs)).sum().to(torch.float32)
    return tp,tn,fp,fn



def prec_rec_spec_neqprec_scores_from_th(y_true:torch.Tensor, y_pred:torch.Tensor,th=0.5) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 2. 0 <= val <= 1
    '''
    assert y_true.ndim == 2
    assert y_pred.ndim == 2
    tp,tn,fp,fn = tp_tn_fp_fn_from_th(y_pred,y_true,th=th)
    epsilon = 0.0000001
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    spec = tn / (tn + fp + epsilon)
    neg_prec = tn / (tn + fn + epsilon)

    result = torch.hstack((precision,recall,spec,neg_prec))
    result.requires_grad = False
    return result

def mcc_with_th(outs,targets,th=0.5):
    '''
    Assumes y_pred is output from sigmoid
    '''
    assert targets.ndim == 2
    assert outs.ndim == 2
    y_pred = (outs > 0.5).to(torch.float32)
    tp,tn,fp,fn = tp_tn_fp_fn_from_th(y_pred,targets,th=th)
    score = mcc_from_xps(tp,tn,fp,fn)
    return score

def f1_from_th(y_true:torch.Tensor, y_pred:torch.Tensor,th=0.5) -> torch.Tensor:
    tp,tn,fp,fn = tp_tn_fp_fn_from_th(outs=y_pred,targets=y_true,th=th)
    epsilon = 0.0000001
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = False
    return f1