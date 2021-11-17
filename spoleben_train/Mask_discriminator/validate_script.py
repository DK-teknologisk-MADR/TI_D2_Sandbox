from pytorch_ML.networks import Classifier , Classifier_Effnet
from efficientnet_pytorch import EfficientNet
from spoleben_train.Mask_discriminator.dataset import Spoleben_Mask_Dataset,Mask_Dataset_Train
import torch
from matplotlib import pyplot
from cv2_utils.cv2_utils import checkout_imgs
from torch.utils.data import DataLoader
import pytorch_ML.validators
from pytorch_ML.trainer import Trainer
import os.path as path
from pytorch_ML.validators import mcc_score,mcc_with_th,f1_from_th
import os
from cv2_utils.cv2_utils import *
from torch.utils.data.dataset import Subset
import torch.nn as nn
from detectron2_ML.data_utils import get_file_pairs
import numpy as np
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

pil_im = Image.open('data/empire.jpg', 'r')
imshow(np.asarray(pil_im))
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_ML.validators import f1_score,f1_score_neg,mcc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau , ExponentialLR
from torch.optim import SGD,Adam
import albumentations as A
from pytorch_ML.model_tester_mask import Model_Tester_Mask
from pytorch_ML.networks import Classifier_Effnet
from pytorch_ML.validators import f1_score_neg, prec_rec_spec_neqprec_scores
data_dir = '/pers_files/spoleben/spoleben_09_2021/spoleben_masks_expert/' #INSERT HERE
output_dir = f"{data_dir}output/output2"
#os.makedirs(output_dir)
splits = ['train','val']
splits = {split : split for split in splits}
mask_nr = 4
size = (450,450)

model = Classifier_Effnet(device = 'cuda',model_name = "efficientnet-b2",in_channels=mask_nr + 3,num_classes =mask_nr)
dt_val = Spoleben_Mask_Dataset(data_dir=path.join(data_dir,splits['val']),mask_nr=mask_nr,size=(450,450))
tester = Model_Tester_Mask(net=model,path=os.path.join(output_dir,"best_model.pth"),device='cuda:1')
bs=10
loader = DataLoader(dt_val,batch_size=1)
targets_ls,out_ls = [],[]
samples = []
fun=f1_from_th
aggregate_device = 'cuda:1'
for batch,targets in loader:
    batch = batch.to('cuda:1')
    target_batch = targets.to('cuda:1')
    out_batch = tester.get_evaluation(batch)
    targets_ls.append(target_batch)
    out_ls.append(out_batch)
    for i in range(1):
        samples.append((batch[i].to('cpu').numpy(),targets[i].to('cpu').numpy(),out_batch[i].to('cpu').numpy()))
sample = samples[2]
img = sample[0][:3]
img = (255*np.transpose(img,(1,2,0))).astype(np.uint8)
masks = [sample[0][i] for i in range(3,7)]
ol = put_mask_overlays(img,masks,colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,255)])
print("true: ",sample[1],"pred: ",1.*(sample[2]>0.5) )
imshow(ol)

outs = torch.cat(out_ls, 0)  # dim i,j,: gives out if j= 0 and target if j = 1
targets = torch.cat(targets_ls, 0)  # dim i,j,: gives out if j= 0 and target if j = 1
index_to_take = torch.argmin(1*(outs>0.5),axis=1)
did_take = torch.tensor([targets[i,index_to_take[i]] for i in range(len(outs))])
would_take = targets[:,0]

scores = []
precs = []
recs = []
specs = []
neg_precs = []

ths = np.arange(0.15,1,0.05)
for th in ths:
    score=fun(y_true = targets,y_pred = outs,th=th)
    prec,rec,spec,neg_prec = prec_rec_spec_neqprec_scores_from_th(targets,outs,th)
    score = score.to('cpu')
    scores.append(score)
    precs.append(prec)
    recs.append(rec)
    specs.append(spec)
    neg_precs.append(neg_prec)
    print("prec :",prec,"rec: ",rec,"f1: ",score)
