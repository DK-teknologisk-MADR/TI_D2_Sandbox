import cv2
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
from cv2_utils.cv2_utils import *
from pytorch_ML.kcentermetric import KCenterMetric
from pytorch_ML.Kmeans import Kmeans_Cities
nr_of_classes = 10
#target_tr = lambda x : x if x<nr_of_classes else nr_of_classes




img_dir = f'/pers_files/encoder_TEST/caltech_pics'
out_dir = path.join(img_dir,"out")



out_dir_collage = path.join(out_dir,"collage")
out_dir_encoder = path.join(out_dir,"encoder")
os.makedirs(out_dir,exist_ok=True)
os.makedirs(out_dir_encoder,exist_ok=True)
os.makedirs(out_dir_collage,exist_ok=True)

cluster_nr = 500
transforms = [A.ColorJitter(hue=0.05),
              A.ShiftScaleRotate(p=0.25, scale_limit=0.00),
              A.HorizontalFlip(p=0.5),
              A.Resize(240, 240),
              A.RandomCrop(height=230, width=230, p=0.25),
              A.Resize(240, 240),
              ]
tr_train = A.Compose(transforms)

class AutoEncoder_Dataset(Dataset):
    def __init__(self,root,tr : A.NoOp = None , tr_train : A.NoOp = None,normalize = True):
        self.tr = tr
        self.tr_resize = A.Resize(240, 240)
        self.tr_train = tr_train
        self.tr_ts = ToTensor()
        self.pic_names = os.listdir(root)
        self.pic_names = [pic_name for pic_name in self.pic_names if pic_name.endswith(".jpeg") or pic_name.endswith(".jpg")]
        self.pic_fps = [path.join(root,pic_name) for pic_name in self.pic_names]
        self.normalize = normalize
        self.has_been_normalized = False
        self.tr_norm = A.Normalize(mean=(0,0,0),std=(1,1,1))
        self.set_normalization()
    def set_normalization(self):
        count = len(self) * 240 ** 2
        psum = torch.zeros(3)
        psum_sq = torch.zeros(3)

        for item in range(len(self)):
            img = self.__getitem__(item)[0]
            psum += img.sum(axis=(1,2))
            psum_sq += (img**2).sum(axis=(1,2))

        # mean and std
        total_mean = psum / count
        total_var = psum_sq / count - total_mean ** 2
        total_std = torch.sqrt(total_var)
        total_mean,total_std = tuple(float(x) for x in total_mean), tuple(float(x) for x in total_std)
        self.tr_norm = A.Normalize(mean=total_mean,std = total_std)

    def get_rgb_img(self,item):
        img = cv2.imread(self.pic_fps[item])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, item):
        img = self.get_rgb_img(item)
        if self.tr_train is not None:
            img = self.tr_train(image=img)['image']
        if self.tr is not None:
            img = self.tr(image=img)['image']
        img = self.tr_resize(image=img)['image']
        if self.normalize:
            img = self.tr_norm(image=img)['image']
        img_ts = self.tr_ts(img)

        return img_ts,img_ts

    def __len__(self):
        return len(self.pic_names)

dataset = AutoEncoder_Dataset(root=img_dir)

train_indices,test_indices = [],[]
for i in range(len(dataset)):
    if i % 10 == 0:
        test_indices.append(i)
    else:
        train_indices.append(i)
data_train,data_test = Subset(dataset,train_indices),Subset(dataset,test_indices)

#output_dir = "/pers_files/encoder_TEST"




bb = ConvAutoencoder(3,16,device='cuda:0')


class RMSE_score():
    def __init__(self):
        self.loss = nn.MSELoss()
    def __call__(self, x,y):
        return -self.loss(x,y)**0.5



loss_fun = nn.MSELoss()
val_fun = RMSE_score()
optimizer = Adam(params = bb.parameters(),lr=0.0001)
scheduler = ExponentialLR(optimizer=optimizer,gamma=1)
trainer = Trainer(dt=data_train,net=bb, optimizer = optimizer, scheduler = scheduler,loss_fun = loss_fun, max_iter= 5*10**4, output_dir =out_dir_encoder,eval_period=100, print_period=50,bs=32,dt_val = data_test,dt_wts = None,fun_val =val_fun , val_nr = None,add_max_iter_to_loaded = False,gpu_id = 0,unsqueeze_ys=False)
trainer.train()

bb.load_state_dict(torch.load(path.join(out_dir_encoder,'best_model.pth'))['state_dict'])
bb.eval()
print("evaluating through encoder...")
dl = DataLoader(dataset=data_train,batch_size=64,pin_memory=True,shuffle = False)
dl = iter(dl)
results = []
for xs,ys in dl:
    with torch.no_grad():
        ts = bb.encoder(xs.to('cuda:0'))
        bs,c,h,w=ts.shape
        ts = ts.reshape(bs,-1)
        results.append(ts)

compressed_data = torch.cat(results, dim=0)
print("shape of encoded data", compressed_data.shape)
compressed_data = compressed_data.to('cpu').numpy()

print("Kmeans")
fitter = Kmeans_Cities(compressed_data)
#fitter = KCenterMetric(compressed_data)
city_indices , cities = fitter.fit(cluster_nr)
#counts = np.unique(fitter.labels,return_counts = True)
counts = np.unique(fitter.cluster_model.labels_,return_counts = True)
size = (500,500)
np.argsort(counts)
#fitter.labels = fitter.cluster_model.labels_
#for i,city in enumerate(cities):
#    is_label_in_cluster_i = labels_of_pics == i
#    dists = fitter.get_distances(data_1=z[is_label_in_cluster_i], data_2 = city)
#    min_dist_indices = np.argsort(dists)
#    if frequency_of_labels>0:
#        min_dist_indices = min_dist_indices[:frequency_of_labels[:i]]
#        min_dist_indices = min_dist_indices.tolist()
#    indices.extend(min_dist_indices)
#city_indices = indices

for city_label, city_index in enumerate(city_indices):
    fp = data_train.dataset.pic_fps[city_index]
    name =data_train.dataset.pic_names[city_index]
    shutil.copy(fp,path.join(out_dir,name[:-4] + "_" + str(city_index) + ".jpg"))
    #make collage
    img = data_train.dataset.get_rgb_img(city_index)
    indices_in_city = np.arange(len(compressed_data))[fitter.labels == city_label]
    nearest_indices = fitter.get_nearest_points(pt = compressed_data[city_index],data=compressed_data[pt_in_city],nr=9)[0]

    pictures = [cv2.resize(data_train.dataset.get_rgb_img(ind), dsize=(size[1], size[0])) for ind in nearest_indices]
    for i in range(9-len(pictures)):
        pictures.append(np.zeros_like(pictures[0]))
    pictures = np.array(pictures)
    pictures = pictures.reshape((3,3,size[0],size[1],3))
    pic_rows = [np.hstack(picture_row) for picture_row in pictures]
    collage = np.vstack(pic_rows)
    cv2.imwrite(path.join(out_dir_collage,name[:-4] + "_" + str(city_index) + ".jpg"),collage[:,:,::-1])
