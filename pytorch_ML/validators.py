import torch








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
#    spec = tn / (tn + fp + epsilon)
#    neg_prec = tn / (tn + fn + epsilon)

    result = torch.hstack((precision,recall))
    result.requires_grad = False
    return result

def f1_score(y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
    prec_rec = prec_rec_scores(y_true,y_pred)
    epsilon = 1e-7
    f1 = 2* (prec_rec[0]*prec_rec[1]) / (prec_rec[0] + prec_rec[1] + epsilon)
    f1.requires_grad = False
    return f1

def f1_score_neg(y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
    print(y_true.shape)
    y_true = torch.abs(1-y_true)
    print(y_true.shape)

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
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
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