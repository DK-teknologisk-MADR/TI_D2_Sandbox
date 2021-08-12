import json

import cv2
import os
import random
import re
from detectron2_ML.data_utils import get_file_pairs, get_data_dicts
import detectron2.data.transforms as T
import torch
import numpy as np

splits = ['train','val','test']
data_dir = "/pers_files/Combined_final/to_crop"
#data_dir = "/pers_files/test_set"
base_output_dir = f'{data_dir}/output'
model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"
new_image_dir = "/pers_files/Combined_final/cropped"
for split in splits:
    os.makedirs(os.path.join(new_image_dir,split),exist_ok=True)
def get_pairs(split):
    files = os.listdir(os.path.join(data_dir,split))
    jpg_fronts = [x.split(".")[0] for x in files if x.endswith(".jpg")]
    data_pairs = {x : [x+".jpg" , x + ".json"] for x in jpg_fronts if  x + ".json" in files}
    return data_pairs
train_pairs = get_pairs('train')
val_pairs = get_pairs('val')
regex_string = r'202[0-9]-'

year_finder = re.compile(regex_string)
def partition_pairs_by_year(data_pairs):

    data_pairs_2020, data_pairs_2021 = {}, {}
    for front,files in data_pairs.items():
        ma = year_finder.search(front)
        if ma:
            if ma.group()=='2020-':
                data_pairs_2020[front] = files
            elif ma.group()=='2021-':
                data_pairs_2021[front] = files
            else:
                print("warning: unknown matched year")
                print(front,ma.group())

        else:
            print("warning: unknown year")
            print(front)
    return data_pairs_2020,data_pairs_2021


def unflatten_polygon_ls(flat_poly):
    assert len(flat_poly) % 2 == 0
    return np.array(flat_poly).reshape(len(flat_poly)//2,2)

def coco_polys_to_ls_of_arrays(annotations):
    poly_list = [annotation['segmentation'][0] for annotation in annotations]
    polygons = [unflatten_polygon_ls(flat_poly) for flat_poly in poly_list]
    return polygons

#data_pairs_2020 , data_pairs_2021 = partition_pairs_by_year(train_pairs)
#for front,values in get_pairs('train').items():
#    regex_string = r'2021-04'
#    double_case_finder = re.compile(regex_string)
#    split = 'train'
#    data_pairs_2021 = {key : val for key,val in data_pairs_2021.items() if not double_case_finder.search(key)}
#    x = cv2.imread(os.path.join(data_dir,split,front+".jpg"))
#    cv2.imshow("window",x[200:720,50:950])
#    cv2.waitKey()
#_ , data_pairs_val = partition_pairs_by_year(val_pairs)
odd_crops = []
crop_values = 40,200,910,530,
def crop_routine():
    trans_split1 = T.CropTransform(63, 512, 898, 512, orig_h=1024, orig_w=1024)
    trans_split2 = T.CropTransform(63,0, 898, 512, orig_h=1024, orig_w=1024)
    resizer = T.ResizeTransform(512,898,530,910)
    trans = T.CropTransform(crop_values[0], crop_values[1], crop_values[2], crop_values[3], orig_h=1024, orig_w=1024)

    trans_split1 = T.TransformList([trans_split1,resizer])
    trans_split2 = T.TransformList([trans_split2,resizer])
#    trans = T.CropTransform(crop_values[0], crop_values[1], crop_values[2], crop_values[3], orig_h=1024, orig_w=1024)
    for split in splits:
        d2_data = get_data_dicts(data_dir,split)
        for file_dict in d2_data:
            img = cv2.imread(file_dict['file_name'])
            with open(file_dict['file_name'][:-4] + ".json") as fp:
                json_dict = json.load(fp)
                if file_dict['file_name'].find("2021-04") == -1: #single case
                    poly_arrs = coco_polys_to_ls_of_arrays(file_dict['annotations'])
                    new_polys = trans.apply_polygons(poly_arrs)
                    new_img = trans.apply_image(img)
                    print(os.path.basename(file_dict['file_name']))
                    print("new poly_len", len(new_polys))
                    print("old poly len", len(json_dict['shapes']))
                    if len(new_polys) == len(json_dict['shapes']):
                        #
                        #     isClosed = True
                        #     # Blue color in BGR
                        #     color = (255, 0, 0)
                        #
                        #     # Line thickness of 2 px
                        #     thickness = 2
                        #
                        #     # Using cv2.polylines() method
                        #     # Draw a Blue polygon with
                        #     # thickness of 1 px
                        #     new_polys_cv = [poly.astype('int32') for poly in new_polys]
                        #     image = cv2.polylines(new_img, new_polys_cv,
                        #                           isClosed, color, thickness)
                        #     cv2.imshow('image', image)
                        #     cv2.imshow('image', img)
                        #     cv2.waitKey()
                        # for i,poly in enumerate(new_polys):
                        #     json_dict['shapes'][i]['points'] = new_polys[i].tolist()
                        #            print(json_dict['shapes'][i]['points'])
                        # get new json file name:
                        front_jpg = os.path.basename(file_dict['file_name'])
                        front_json = front_jpg[:-4] + ".json"
                        os.path.join(new_image_dir, split, )
                        new_json_name = os.path.join(new_image_dir, split, front_json)
                        print(new_json_name)
                        cv2.imwrite(os.path.join(new_image_dir, split, os.path.basename(file_dict['file_name'])),
                                    new_img)
                        with open(new_json_name, "w+") as fp:
                            json.dump(obj=json_dict, fp=fp, indent=3)
                else: #double case
                    poly_arrs = coco_polys_to_ls_of_arrays(file_dict['annotations'])
                    new_polys = [trans_split1.apply_polygons(poly_arrs),trans_split2.apply_polygons(poly_arrs)]
                    new_imgs = [trans_split1.apply_image(img),trans_split2.apply_image(img)]
                    print(os.path.basename(file_dict['file_name']))
                    if len(new_polys[0]) + len(new_polys[1]) == len(json_dict['shapes']):
                        print("good to go")

                        #
                        #     isClosed = True
                        #     # Blue color in BGR
                        #     color = (255, 0, 0)
                        #
                        #     # Line thickness of 2 px
                        #     thickness = 2
                        #
                        #     # Using cv2.polylines() method
                        #     # Draw a Blue polygon with
                        #     # thickness of 1 px
                        #     new_polys_cv = [poly.astype('int32') for poly in new_polys]
                        #     image = cv2.polylines(new_img, new_polys_cv,
                        #                           isClosed, color, thickness)
                        #     cv2.imshow('image', image)
                        #     cv2.imshow('image', img)
                        #     cv2.waitKey()
                        # for i,poly in enumerate(new_polys):
                        #     json_dict['shapes'][i]['points'] = new_polys[i].tolist()
                #            print(json_dict['shapes'][i]['points'])
                        #get new json file name:
                        front_jpgs = [os.path.basename(file_dict['file_name'])[:-4] + direction + ".jpg" for direction in ["upper","lower"]]
                        front_jsons = [front_jpg[:-4] + ".json" for front_jpg in front_jpgs]
                        os.path.join(new_image_dir,split,)
                        new_json_names = [os.path.join(new_image_dir,split,front_json) for front_json in front_jsons]
                        print(new_json_names)
                        for i in range(2):
                            cv2.imwrite(os.path.join(new_image_dir,split,front_jpgs[i]),new_imgs[i])
                            with open(new_json_names[i],"w+") as fp:
                                json.dump(obj=json_dict,fp=fp,indent=3)
crop_routine()

fp = "/pers_files/Combined_final/to_crop/train/robotcell_all1_color_2021-04-08-10-11-34.jpg"
img = cv2.imread(fp)[:515,47:1024-47]
cv2.imshow("double",img)
cv2.waitKey()

def ts_to_rgb(ts):
    return ts.permute()
front = random.sample(list(data_pairs_2021),1)[0]
front = random.sample(list(data_pairs_2021),1)[0]
x = cv2.imread(os.path.join(data_dir,split,front+".jpg"))
x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
trans = T.CropTransform(40,190,910,530,orig_h=1024,orig_w=1024)
resize = T.Resize(512,512)
bright = T.RandomBrightness(0.85,1.2)
light = T.RandomLighting(0.85,1.2)
aug = T.AugmentationList([trans,bright,light])
input = T.AugInput(x[:,:,[2,1,0]])
#trans = T.CropTransform(40,190,910,530)
new_img = aug(input)
new_img = new_img.apply_image(x)
new_img = new_img.astype('uint8')
new_img = cv2.cvtColor(new_img,cv2.COLOR_RGB2BGR)
cv2.imshow("lul",new_img)
cv2.waitKey()


