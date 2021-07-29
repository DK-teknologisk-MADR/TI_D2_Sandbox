import torch
from torchvision import transforms
from torch import sigmoid
import numpy as np
from pytorch_ML.Model_Tester import Model_Tester


class Model_Tester_Mask(Model_Tester):
    def __init__(self,net,path,trs_x = [],device='cuda:0'):
        self.trs_x = trs_x
        if trs_x:
            self.trs_x_comp = transforms.Compose(trs_x)
        self.device = device
        super().__init__(net=net,path_to_save_file=path,device=device)


    def get_evaluation(self,picture,need_sig = True):
        '''
        assumes CxHxW picture
        '''
        if torch.is_tensor(picture):
            picture = picture.to(self.device)
        else:
            picture = torch.tensor(picture,device=self.device,requires_grad=False)
        with torch.no_grad():
            if self.trs_x:
                picture = self.trs_x_comp(picture)
            out = self.net(picture)
            if need_sig:
                out = sigmoid(out)
            res = out
        return res
#move to script in the end:
#-----------------------------
def normalize_ious(arr):
    x0 = 0.8
    x1 = 0.94
    arr = (arr - x0) * 1/(x1-x0)
    return np.where(arr<0,0,
             np.where(arr>1,1,arr))

def iou_inverse(ts):
    return (0.94-0.8)*ts + 0.8
 #----------------

#for i in range(len(df)):
#    df.loc[i,'iou_pred'] = df.loc[i,'iou_pred'][0][0]
#    df.loc[i,'diff'] = df.loc[i,'diff'][0][0]