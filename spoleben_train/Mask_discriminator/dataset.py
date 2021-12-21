import numpy as np
import cv2
import pandas
from data_and_file_utils.file_utils import get_file_with_ending
from cv2_utils.cv2_utils import *
import glob, os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler,Subset
import json
from torchvision.transforms import ToTensor
import albumentations as A
import os.path as path
import re
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
    def __init__(self,data_dir,mask_nr,csv_name = None,size = (450,450),mask_cols = None,with_depth = False):
        if csv_name is None:
            csv_name = get_file_with_ending(data_dir,'.csv')
        csv_fp = path.join(data_dir,csv_name)
        print(csv_fp)
        self.with_depth = with_depth
        self.df = pd.read_csv(csv_fp,sep=',')
        is_null = pd.isnull(self.df).all(axis=1)
        self.df = self.df[np.logical_not(is_null)]
        self.df.reset_index(inplace=True)
        self.df['pic_ID'] = self.df['index'].astype('int')
        self.df.index.astype('int')
        print(self.df)
        self.size = size
        self.mask_nr = mask_nr
        self.ch_nr = mask_nr + self.with_depth + 3
        self.mask_start_index = 3 + self.with_depth
        self.totensor = ToTensor()
        self.data_dir = data_dir
        if mask_cols is None:
            self.mask_cols = [f'mask{i}' for i in range(1,mask_nr + 1)]
        else:
            self.mask_cols = mask_cols
        self.data_path_dict = self.get_file_paths_from_df()

    def get_file_paths_from_df(self):
        data_dict_by_id = {}
        data_dict_by_item = {}

        rex = re.compile('_id(\d+)')
        re_type = re.compile('(mask\d+)|(color)|(depth)')

        files = os.listdir(self.data_dir)
        for file in files:
            if not file.endswith(".csv"):
                match = rex.search(file)
                id = int(match[1])
                match = re_type.search(file)
                key = match[0]
                if key.startswith("mask"):
                    mask_id = int(key[-1])
                    mask_id -=1
                    key = key[:-1] + str(mask_id)
                    assert mask_id>-1 and mask_id < self.mask_nr, f'mask_id is {mask_id}'
                if key == 'color':
                    key = 'raw'
                if id in data_dict_by_id:
                    data_dict_by_id[id][key] = path.join(self.data_dir,file)
                else:
                    data_dict_by_id[id] = {key : path.join(self.data_dir,file)}
        for id,data in data_dict_by_id.items():
            for i in range(0,self.mask_nr):
                if not f'mask{i}' in data:
                    data[f'mask{i}'] = None
        for item in range(len(self.df)):
            id = self.df.loc[item,'pic_ID']
            data_dict_by_item[item] = data_dict_by_id[id]
        return data_dict_by_item
    
    


    def __len__(self):
        return len(self.df)

    def get_item_as_ndarray(self,item):
        labels = np.array(self.df.loc[item, self.mask_cols],dtype = np.float)
        paths = self.data_path_dict[item]
        imgxc = np.zeros((self.size[0],self.size[1],self.ch_nr),dtype=np.uint8)
        imgxc[:,:,:3] = imread_as_rgb(paths['raw'])
        if self.with_depth:
            mask = cv2.imread(paths['depth'],cv2.IMREAD_GRAYSCALE)
            imgxc[:, :, 3] = cv2.threshold(mask,128,255,cv2.THRESH_BINARY)[1]
        for i in range(self.mask_nr):
            mask_path = paths[f'mask{i}']
            if mask_path is not None:
                mask = cv2.imread(paths[f'mask{i}'],cv2.IMREAD_GRAYSCALE)
                imgxc[:, :, self.mask_start_index + i] = cv2.threshold(mask,128,255,cv2.THRESH_BINARY)[1]
            else:
                imgxc[:, :, self.mask_start_index + i] = 0
#        checkout_imgs([imgxc[:,:,3],imgxc[:,:,4],imgxc[:,:,5]])
        return imgxc,labels

    def __getitem__(self, item):
        imgxc,labels = self.get_item_as_ndarray(item)
        tsxc = self.totensor(imgxc)
        return tsxc,labels




class Mask_Dataset_Train(Spoleben_Mask_Dataset):
    def __init__(self,data_dir,mask_nr,csv_name = None,augs = None,**kwargs):
        super().__init__(data_dir=data_dir,mask_nr = mask_nr,csv_name=csv_name,**kwargs)
        self.augs = augs


    def mask_shuffle_procedure(self,image, masks, labels):
        '''
        Keeps depth mask fixed, but shuffles the remaining masks and their respective labels
        '''
        indices = np.random.choice(np.arange(0,self.mask_nr), self.mask_nr, replace=False)
        labels = labels[indices]
        masks = [masks[index] for index in indices]
        return image,masks,labels


    def mask_delete_procedure(self,image,masks,labels,p_nrs : np.ndarray = np.array([0.,0.,0.1,0.15,0.75])):
        '''
        Will randomly delete masks. There is p_nrs[i] probability of keeping i masks. "deleted mask" will have label 0 and np.zeros as mask
        '''
        p_nrs = p_nrs / p_nrs.sum()
        assert len(p_nrs) == self.mask_nr + 1 , "each index in p_nrs corresponds to probability of keeping i masks"
        nr_to_keep = np.random.choice(np.arange(0,self.mask_nr + 1),p = p_nrs)
        if nr_to_keep < self.mask_nr:
            indices_to_delete = np.random.choice(np.arange(0,self.mask_nr), self.mask_nr - nr_to_keep, replace=False)
            for mask_index in indices_to_delete:
                masks[mask_index][:,:] = 0
                labels[mask_index] = 0
        return image,masks,labels


    def get_item_as_ndarray(self,item):
        img5c,labels = super().get_item_as_ndarray(item)
        masks = [mask.squeeze(2) for mask in np.split(img5c[:, :, self.mask_start_index: ], indices_or_sections=self.mask_nr + self.with_depth, axis=2)]

        image,masks,labels = self.mask_shuffle_procedure(image= img5c[:, :, :3],masks= masks,labels = labels)
        image,masks,labels = self.mask_delete_procedure(image= img5c[:, :, :3],masks= masks,labels = labels)

        if self.augs is not None:
            img5c_aug_dict = self.augs(image=image, masks=masks,labels=labels)
            image,masks,labels = img5c_aug_dict['image'], img5c_aug_dict['masks'], img5c_aug_dict['labels']
        ls_to_img = masks.copy()
        ls_to_img.insert(0,image)
        img5c_aug = np.dstack(ls_to_img)
        return img5c_aug , labels


dt = Mask_Dataset_Train(data_dir='/pers_files/spoleben/spoleben_09_2021/spoleben_masks_expert/train',mask_nr = 4,size=(450,450))
dt.get_file_paths_from_df()
#for img,y in x:
#    print(img,y)

#cv2.imread('/pers_files/spoleben/spoleben_09_2021/spoleben_masks_expert/kinect_20210916_094132_color__ID83crop1_c0x.jpg')
#np.array(dt.df.loc[0, dt.mask_cols],dtype='float')

