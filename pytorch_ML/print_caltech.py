import torchvision
import os
import torch
import scipy
import os.path as path
import shutil
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_ML.networks import Classifier,Classifier_Multi_Effnet,Encoder,ConvAutoencoder
from torch.utils.data import DataLoader,Dataset
from pytorch_ML.data_utils import get_balanced_class_weights_from_dataset
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import cv2
import shutil
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_ML.validators import f1_score,f1_score_neg,mcc_score, F1_Score_Osc
from detectron2_ML.pruners import SHA
from torch.utils.data import Dataset
import matplotlib
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
matplotlib.use('tkAgg')
from torchvision.transforms import ToTensor
from pytorch_ML.trainer import Trainer
import albumentations as A
from torch.optim import SGD,Adam
from torch.utils.data import Subset
output_dir = "/pers_files/encoder_TEST/caltech_pics"
class my_transform():
    def __init__(self):
        transforms = [
            A.Resize(240, 240),
                      ]
        self.tr = A.Compose(transforms)

    def __call__(self,img):
        img = np.array(img)
        img = self.tr(image=img)['image']
        return img
tr = my_transform()

dataset = torchvision.datasets.Caltech101(root="/pers_files/caltech/",transform=tr,download=False)

color_pics = []
for i,data in enumerate(dataset):
    img,y = data
    if img.ndim == 3:
        color_pics.append(i)
dataset =Subset(dataset,color_pics)
os.makedirs(output_dir,exist_ok=True)
for i,data in enumerate(dataset):
    x,y = data
    x_bgr = x[:,:,::-1]
    print(x_bgr.shape)
    cv2.imwrite(path.join(output_dir,f"pic{i}.jpg"),x_bgr)
