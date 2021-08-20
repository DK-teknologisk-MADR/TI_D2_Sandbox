import numpy as np
import cv2
import matplotlib
import torchvision.transforms.functional
from cv2_utils.cv2_utils import centroid_of_mask_in_xy
matplotlib.use('TkAgg')
import detectron2.data.transforms as T
from torchvision.transforms import ToTensor,Normalize,Compose,RandomAffine,ColorJitter
from torch.nn import Conv2d
import torch
class CropAndResize(T.Augmentation):
    def __init__(self, scale_range,shift_range=[0,0]):
        self.scale_range = scale_range
        self.shift_range = shift_range

    def __call__(self,aug_input):
        oldx,oldy,oldc = aug_input.image.shape
        scaler = T.RandomExtent(self.scale_range,self.shift_range)(aug_input)
        resizer = T.Resize((oldx,oldy))(aug_input)
        return T.TransformList([scaler,resizer])


def centralize_img_wo_crop(img,fixed_translation = None):
    height,width,ch = img.shape
    non_zero_coords = np.array(np.nonzero(img))
    flag = False

    if fixed_translation is None:
        center = non_zero_coords.mean(axis=1)
        h_min,w_min = non_zero_coords.min(axis=1)[:2]
        h_max,w_max = non_zero_coords.max(axis=1)[:2]
        t_1 = width // 2-center[1]
        t_2 = height // 2-center[0]
        if t_1 + w_min<0:
            t_1 = 0
            flag = True
        elif t_1+w_max>width:
            t_1 = width-w_max
            flag = True

        if t_2 + h_min < 0:
            t_2 = 0
            flag = True

        elif t_2 + h_max > height:
            t_2 = height - h_max
            flag = True
    else:
        t_1,t_2 = fixed_translation

    T = np.float32([[1, 0, t_1], [0, 1, t_2]])
    img_translation = cv2.warpAffine(img, T, (width, height))
    return img_translation,(t_1,t_2),flag

def get_largest_component_and_centroid(img):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
    largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    output = np.where(output == largest_component_label,1,0)
    return output,centroids



class PreProcessor():
    def __init__(self,crop,resize_dims,pad,mean,std):
        self.to_ts = ToTensor()
        self.norm = Normalize(mean=mean,std=std)
        self.torchvision_transforms = Compose([self.to_ts,self.norm])
        self.crop = crop
        self.resize_dims = resize_dims
        self.conv = Conv2d(1, 1, (pad, pad), bias=False,padding=(pad//2,pad//2),).requires_grad_(False) #to be put in predictor class
        self.conv.weight[0] = torch.ones_like(self.conv.weight[0],device='cpu')

    def preprocess(self,img,raw_mask):
        '''
        BGR_IMG
        '''
        mask = np.asarray(raw_mask>0,dtype='uint8')
        mask, centroids, area, discard_flag = self.get_component_analysis(mask)
        height, width, ch = img.shape
        t_1 = width // 2 - centroids[0]
        t_2 = height // 2 - centroids[1]
        T = np.float32([[1, 0, t_1], [0, 1, t_2]])
        mask_center = cv2.warpAffine(mask, T, (width, height))[self.crop[0][0]: self.crop[0][1], self.crop[1][0]: self.crop[1][1]]
        img_center = cv2.warpAffine(img, T, (width, height))[self.crop[0][0]: self.crop[0][1], self.crop[1][0]: self.crop[1][1]]
        mask_center = torch.tensor(mask_center,dtype=torch.float).unsqueeze(0)
        mask_padded = self.conv(mask_center.unsqueeze(0)).squeeze(0)
        mask_padded = mask_padded.bool().long()
        img_ts = self.torchvision_transforms(img_center)
        img_masked = (img_ts * mask_padded)
        full_output = torch.vstack([img_masked,mask_center])
        full_output = torchvision.transforms.functional.resize(full_output,self.resize_dims)
        return full_output

    def get_component_analysis(self, mask):
        '''
        returns:mask of biggest component
        centroids in format (x from left, y from top)
        largest area in pixel count
        discard_flag hints whether its too weird to keep.
        '''
        discard_flag = False
        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
        # centroids in x from left, y from top.
        largest_components = stats[1:, 4].argsort()[::-1] + 1
        largest_area = stats[largest_components[0], 4]
        proportions = stats[1:, 4] / largest_area
        if np.any(np.logical_and(proportions > 0.2, proportions < 1)):
            discard_flag = True
        mask = np.where(labels == largest_components[0], 1, 0).astype('uint8')
        centroids = centroids[largest_components[0]]
        return mask, centroids, largest_area, discard_flag


class PreProcessor_Box_Crop():
    def __init__(self, crop,resize_dims, pad, mean, std):
        self.to_ts = ToTensor()
        self.norm = Normalize(mean=mean, std=std)
        self.torchvision_transforms = Compose([self.to_ts, self.norm])
        self.crop = crop
        self.pad = pad
        self.resize_dims = tuple(resize_dims)

    def preprocess(self, img, raw_mask):
        '''
        BGR_IMG
        '''
        mask = np.asarray(raw_mask > 0, dtype='uint8')
        mask, centroids, area, discard_flag = self.get_component_analysis(mask)
        col,row,w,h = cv2.boundingRect(cv2.findNonZero(mask))
        rect_crop_dims = max(row-self.pad,0),min(row+h+self.pad,mask.shape[0]),max(col-self.pad,0),min(col+w+self.pad,mask.shape[1])
        cropped_out_img = np.zeros_like(img)
        cropped_out_img[rect_crop_dims[0]:rect_crop_dims[1],rect_crop_dims[2]:rect_crop_dims[3]] = img[rect_crop_dims[0]:rect_crop_dims[1],rect_crop_dims[2]:rect_crop_dims[3]]
        height, width, ch = img.shape
        t_1 = width // 2 - centroids[0]
        t_2 = height // 2 - centroids[1]
        T = np.float32([[1, 0, t_1], [0, 1, t_2]])
        mask_center = cv2.warpAffine(mask, T, (width, height))[self.crop[0][0]: self.crop[0][1],
                      self.crop[1][0]: self.crop[1][1]]
        img_center = cv2.warpAffine(cropped_out_img, T, (width, height))[self.crop[0][0]: self.crop[0][1],
                     self.crop[1][0]: self.crop[1][1]]
        img_full = np.concatenate([img_center,mask_center[:,:,None]],axis=2)
        img_full = cv2.resize(img_full,(self.resize_dims[1],self.resize_dims[0]))
        img_full[:,:,3]= np.where(img_full[:,:,3]>0,255,0)
        img_full = self.torchvision_transforms(img_full)

        return img_full
#x = PreProcessor_Box_Crop((0,1024,0,1024),(500,500),15,[0,0,0],[1,1,1])

    def get_component_analysis(self, mask):
        '''
        returns:mask of biggest component
        centroids in format (x from left, y from top)
        largest area in pixel count
        discard_flag hints whether its too weird to keep.
        '''
        discard_flag = False
        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
        # centroids in x from left, y from top.
        largest_components = stats[1:, 4].argsort()[::-1] + 1
        largest_area = stats[largest_components[0], 4]
        proportions = stats[1:, 4] / largest_area
        if np.any(np.logical_and(proportions > 0.2, proportions < 1)):
            discard_flag = True
        mask = np.where(labels == largest_components[0], 1, 0).astype('uint8')
        centroids = centroids[largest_components[0]]
        return mask, centroids, largest_area, discard_flag


class PreProcessor_Crop_n_Resize_Box():
    def __init__(self, resize_dims, pad, mean, std):
        self.to_ts = ToTensor()
        self.norm = Normalize(mean=mean, std=std)
        self.torchvision_transforms = Compose([self.to_ts, self.norm])
        self.pad = pad
        self.resize_dims = tuple(resize_dims)

    def preprocess(self, img, raw_mask):
        '''
        BGR_IMG
        '''
        mask = np.asarray(raw_mask > 0, dtype='uint8')
        mask, centroids, area, discard_flag = self.get_component_analysis(mask)
        col,row,w,h = cv2.boundingRect(cv2.findNonZero(mask))
        rect_crop_dims = max(row-self.pad,0),min(row+h+self.pad,mask.shape[0]),max(col-self.pad,0),min(col+w+self.pad,mask.shape[1])
        centroids = row + w//2 , col + h // 2
        cropped_out_img = np.zeros_like(img)
        crop_img = img[rect_crop_dims[0]:rect_crop_dims[1],rect_crop_dims[2]:rect_crop_dims[3]]
        mask = mask[rect_crop_dims[0]:rect_crop_dims[1],rect_crop_dims[2]:rect_crop_dims[3]]
        img_full = np.concatenate([crop_img,mask[:,:,None]],axis=2)
        img_full = cv2.resize(img_full,(self.resize_dims[1],self.resize_dims[0]))
        img_full[:,:,3]= np.where(img_full[:,:,3]>0,255,0)
        img_full = self.torchvision_transforms(img_full)
        return img_full
#x = PreProcessor_Box_Crop((0,1024,0,1024),(500,500),15,[0,0,0],[1,1,1])
    def get_component_analysis(self, mask):
        '''
        returns:mask of biggest component
        centroids in format (x from left, y from top)
        largest area in pixel count
        discard_flag hints whether its too weird to keep.
        '''
        discard_flag = False
        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
        # centroids in x from left, y from top.
        largest_components = stats[1:, 4].argsort()[::-1] + 1
        largest_area = stats[largest_components[0], 4]
        proportions = stats[1:, 4] / largest_area
        if np.any(np.logical_and(proportions > 0.2, proportions < 1)):
            discard_flag = True
        mask = np.where(labels == largest_components[0], 1, 0).astype('uint8')
        centroids = centroids[largest_components[0]]
        return mask, centroids, largest_area, discard_flag

