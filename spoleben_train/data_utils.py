import os
import sys

import cv2
import numpy as np
from detectron2.structures.boxes import BoxMode
from pycocotools.mask import encode




def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax,cmax

def bbox2_xyxy_abs(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return cmin,rmin, cmax,rmax


def sort_by_prefix(fp):
    #file_ls = ["hallo.jpg","hallo.json","hallo_also.jpg","hallibu_dallibu.jpg"]
    file_ls = os.listdir(fp)
    result = {}
    NEW_FRONT = 0
    EXISTING_FRONT = 1
    SMALLER_FRONT = 2

    for file in file_ls:
        print("looping",file)
        front_of_file, ending_of_file = split_by_ending(file)
        category = NEW_FRONT
        smaller_than = []
        larger_than = []
        for front,files in result.items():
            if front.startswith(front_of_file) and front>front_of_file:
                print("appending front",front," to smalelr than ",front_of_file)
                smaller_than.append(front)
            elif front_of_file.startswith(front):
                print("appending front",front," to larger than ",front_of_file)
                larger_than.append(front)
                if len(larger_than)>1:
                    raise ValueError("something is wrong, got ", larger_than, "with ",front,front_of_file)

        if not (smaller_than or larger_than):
           result[front_of_file] = [file]
        elif smaller_than:
            result[front_of_file] = [file]
            for front in smaller_than:
                ls = result.pop(front)
                result[front_of_file].extend(ls)
            print(front_of_file)
        elif larger_than:
            result[larger_than[0]].append(file)

    #sort files
    for files in result.values():
        files.sort()
    return result

def split_by_ending(file_name):
    '''
    like split but adds "" if no ending
    '''

    x = file_name.split(".")
    if len(x)>1:
        return x[-2:]
    else:
        return [x[-1],""]


def load_masks(fp_ls):
    masks = []
    for i,file in enumerate(fp_ls):
        mask = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        mask = mask // 255
#        print("loading masks of shape",mask.shape)
        if mask is None:
            raise ValueError("file",file,"did not seems to exist.")
        masks.append(mask)
    return masks

def load_and_batch_masks(fp_ls):
        masks = load_masks(fp_ls)
        print(masks[0].shape,fp_ls)
        h,w = masks[0].shape
        for i,mask in enumerate(masks):
            if mask.shape != masks[0].shape:
                raise ValueError("mask from file ",fp_ls[i], "are not of the same shape as ",fp_ls[0])
        mask_batch = np.array(masks,dtype='bool')

        return mask_batch


def get_small_masks(masks,small_th):
    sizes = masks.sum(axis=(1,2))
    is_small = sizes < small_th
    return is_small


def percentage_of_masks_in_crop(masks,crop):
    assert masks.ndim == 3
    full = masks.sum(axis=(1,2))
    crop_masks = masks[:,crop[0]:crop[2],crop[1]:crop[3]]
    cropped = masks[:,crop[0]:crop[2],crop[1]:crop[3]].sum(axis=(1,2))
    print(masks.shape)
    print(crop_masks.shape)
    print("full",full,"cropped",cropped)
    if np.any(full == 0):
        print(full == 0)
        raise ValueError("u passed empty mask")
    return cropped / full

def show_mask(img,masks):
    '''
    assume img is uint8, mask is 0/1
    '''
    colors =[
        [0,204,255],
        [255,255,0],
        [255,0,0],
        [0,255,0],
        [0,0,255],
        [128,0,128],
        [255,102,0],
        [255,0,255],
        [255,204,153],
    ]
    for i,mask in enumerate(masks):
        img[mask.astype('bool')] = colors[i % len(colors)]
    cv2.imshow("show_mask",img)



def get_data_dicts_masks(data_dir,split,file_pairs):
    '''
    input:
    list(str) each str refers to a split of dataset. example : split = ['train','val','test']
    If file_pairs is none, file_pairs will be created all files in each data_dir split.
    output: dict withCOCO compliant structure, ready to be registered as a dataset in detectron2
    '''
    dataset_dicts = []
    data_dir_cur = os.path.join(data_dir,split)
    for key,values in file_pairs.items():
        if len(values)!=2:
            raise ValueError(str(values) + "should be of length 2")
    img_id = 0
    for idx,tup in enumerate(file_pairs.items()):
        name, files = tup
        files.sort()
        jpg_name,npy_name = files
        #print(jpg_name)
        #print(cv2.imread(data_dir_cur))
        try:
            height, width = cv2.imread(os.path.join(data_dir_cur,jpg_name)).shape[:2]
        except AttributeError:
            print(f"could not load {os.path.join(data_dir_cur,jpg_name)}")
            sys.exit()
        record = {}
        record["file_name"] = os.path.join(data_dir_cur,jpg_name)
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        mask_file = os.path.join(data_dir_cur,npy_name)
        masks = np.load(mask_file)
        masks_rle = encode(np.asarray(masks.transpose(1,2,0), order="F"))

        print(len(masks_rle))
        print(masks.shape)
        objs = []
        for i in range(masks.shape[0]):
            xmin,ymin,xmax,ymax = bbox2_xyxy_abs(masks[i])
            obj = {
                        "bbox": [xmin,ymin,xmax,ymax],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": masks_rle[i],
                        "category_id": 0,
                    }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


