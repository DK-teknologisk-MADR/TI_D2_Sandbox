import numpy as np
import cv2
import pandas
from data_utils.file_utils import get_file_with_ending
from cv2_utils.cv2_utils import *
import glob, os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler,Subset
import json
import albumentations as A
import os.path as path
from detectron2_ML.data_utils import get_file_pairs
from filet.mask_discriminator.transformations import PreProcessor,PreProcessor_Box_Crop

#import matplotlib
#matplotlib.use('TkAgg')





test_df = pd.DataFrame([[0,1,1,0],[0,0,0,0],[1,1,1,1],[1,1,0,0],[1,0,1,0],[1,1,0,0],[1,0,0,0],[1,1,0,1]], columns=[f'mask{i}' for i in range(4)])
test_df.index = list(range(1,len(test_df) + 1))
test_df.index.name = "pic_ID"
test_df = test_df.melt(ignore_index=False,var_name = "mask",value_name='label')
test_df.reset_index(inplace=True)

#x = A.ColorJitter()
#img = np.random.randint(0,255,(450,450,5)).astype(np.uint8)
#img2 = x(image = img[:,:,:3],masks = [img[:,:,3],img[:,:,4]])

class Spoleben_Mask_Dataset(Dataset):
    def __init__(self,data_dir,csv_name = None,size = (450,450)):
        if csv_name is None:
            csv_name = get_file_with_ending(data_dir,'.csv')
        csv_fp = path.join(data_dir,csv_name)
        self.df = self.df # self.df  = pd.DataFrame([0,1,2,3,4]) #self.df = pd.read_csv(csv_fp,header=True,index_col='index')
        self.df = self.df.melt(ignore_index=False,var_name = "mask",value_name='label')
        self.df.reset_index(inplace=True)
        self.size = size


    def generate_paths(self,int : id) -> dict:
        paths = {'raw' : '',
                 'depth' : '',
                 'mask0' : '',
                 'mask1': '',
                 'mask2': '',
                 'mask3': '',
                 }
        return paths

    def __len__(self):
        return len(self.df)


    def __getitem__(self, item):
        id = self.df.loc[item,'pic_ID']
        label = self.df.loc[item, 'label']
        mask_str = self.df.loc[item,'mask']
        paths = self.generate_paths(id)
        img5c = np.zeros((self.size[0],self.size[1],5))
        img5c[:,:,:3] = imread_as_rgb(paths['raw'])
        img5c[:,:,4] = cv2.imread(paths['depth'],cv2.IMREAD_GRAYSCALE)
        img5c[:,:,5] = cv2.imread(paths['mask0'],cv2.IMREAD_GRAYSCALE)
        return img5c,label

class Mask_Dataset_Train(Spoleben_Mask_Dataset):
    def __init__(self,data_dir,csv_name = None,augs = None):
        super().__init__(data_dir=data_dir,csv_name=csv_name)
        self.augs = augs

    def __getitem__(self, item):
        img5c,label = super().__getitem__(item)
        if self.augs is not None:
            img5c_aug_dict = self.augs(image=img5c[:, :, :3], masks=[img5c[:, :, 3], img5c[:, :, 4]])
            img5c_aug = np.dstack([img5c_aug_dict['image'], img5c_aug_dict['masks'][0], img5c_aug_dict['masks'][1]])
            return img5c_aug,label
        else:
            return img5c , label

    #TODO::FINISH THIS

