import json
import yaml
import cv2
import os
import random
import re
import cv2_utils.cv2_utils
from detectron2_ML.data_utils import get_file_pairs, get_data_dicts
import detectron2.data.transforms as T
import torch
import numpy as np
from copy import deepcopy

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
#train_pairs = get_pairs('train')
#val_pairs = get_pairs('val')
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
def modify_TI_json_metadata_(json_dict,new_name = None):
    json_dict["imageHeight"] =  crop_values[3]
    json_dict["imageWidth"] = crop_values[2]
    if new_name is not None:
        json_dict["imagePath"] = new_name,
    return json_dict

def crop_routine():
    trans_split1 = T.CropTransform(x0=63, y0=512, w=898, h=512, orig_h=1024, orig_w=1024)
    trans_split2 = T.CropTransform(x0=63,y0=0, w=898, h=512, orig_h=1024, orig_w=1024)
    resizer = T.ResizeTransform(h=512,w=898,new_h= 530,new_w=910)
    trans_split1 = T.TransformList([trans_split1,resizer])
    trans_split2 = T.TransformList([trans_split2,resizer])
    trans = T.CropTransform(x0=crop_values[0], y0=crop_values[1], w=crop_values[2], h=crop_values[3], orig_h=1024, orig_w=1024) #for single case

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
                if len(new_polys) == len(json_dict['shapes']):
#                        poly_overlay = cv2_utils.cv2_utils.put_polys(new_img,new_polys)
#                        cv2.imshow("window",poly_overlay)
#                        cv2.waitKey()
                    modify_TI_json_metadata_(json_dict)
                    json_dict['shapes'] =[{} for _ in range(len(new_polys))]
                    for i,poly in enumerate(new_polys):
#                       print(new_polys[i].tolist())
                        json_dict['shapes'][i]['points'] = poly.tolist()
#                            print(json_dict['shapes'][i]['points'])
#                        get new json file name:
                    front_jpg = os.path.basename(file_dict['file_name'])
                    front_json = front_jpg[:-4] + ".json"
                    os.path.join(new_image_dir, split, )
                    new_json_name = os.path.join(new_image_dir, split, front_json)
#                        print(new_json_name)
                    cv2.imwrite(os.path.join(new_image_dir, split, os.path.basename(file_dict['file_name'])),
                                new_img)
                    with open(new_json_name, "w+") as fp:
                        json.dump(obj=json_dict, fp=fp, indent=3)
            else: #double case
                poly_arrs = coco_polys_to_ls_of_arrays(file_dict['annotations'])
                new_poly_groups = [trans_split1.apply_polygons(poly_arrs),trans_split2.apply_polygons(poly_arrs)]
                new_imgs = [trans_split1.apply_image(img),trans_split2.apply_image(img)]
                front_jpgs = [os.path.basename(file_dict['file_name'])[:-4] + direction + ".jpg" for direction in
                              ["lower", "upper"]]
                if len(new_poly_groups[0]) + len(new_poly_groups[1]) == len(json_dict['shapes']):

                    new_jsons = [modify_TI_json_metadata_(deepcopy(json_dict),front_jpgs[i]) for i in range(2)]
                    for i,new_json in enumerate(new_jsons):
                        new_jsons[i]['shapes'] = [{} for _ in range(len(new_poly_groups[i]))]
                    for lu_id,new_polys in enumerate(new_poly_groups):
                        json_dict = new_jsons[lu_id]
                        for i,poly in enumerate(new_polys):
                    #        print(new_jsons[lu_id]['shapes'][i]['points'])

                            new_jsons[lu_id]['shapes'][i]['points'] = poly.tolist()
                  #          print("CHANGED TO")
                  #          print(new_jsons[lu_id]['shapes'][i]['points'])
                    #get new json file name:
                    front_jsons = [front_jpg[:-4] + ".json" for front_jpg in front_jpgs]
                    new_json_names = [os.path.join(new_image_dir,split,front_json) for front_json in front_jsons]

                    plot_polys = [np.array(new_jsons[0]['shapes'][i]['points']) for i in range(len(new_jsons[0]['shapes']))]
                    poly_overlay = cv2_utils.cv2_utils.put_polys(new_imgs[0],plot_polys)
#                    print("CV2 INPUT IS",new_poly_groups[0][0].shape)
#                     cv2.imshow("window",poly_overlay)
#                     cv2.waitKey()
#                     poly_overlay = cv2_utils.cv2_utils.put_polys(new_imgs[1], new_poly_groups[1])
#                     cv2.imshow("window1", poly_overlay)
#                     cv2.waitKey()
                    for i in range(2):
                        cv2.imwrite(os.path.join(new_image_dir,split,front_jpgs[i]),new_imgs[i])
                        with open(new_json_names[i],"w+") as fp:
                            json.dump(obj=new_jsons[i],fp=fp,indent=3)
#                    for i in range(2):
#                        my_img = cv2.imread(os.path.join(new_image_dir,split,front_jpgs[i]))
#                        with open(new_json_names[i],"r+") as fp:
#                            check_dict = json.load(fp=fp)
#                        polys = [np.array(check_dict['shapes'][j]['points']) for j in range(len(check_dict['shapes']))]
#                        poly_overlay = cv2_utils.cv2_utils.put_polys(my_img, polys)
#                        cv2.imshow(f"window_after_load{i}", poly_overlay)
#                        cv2.waitKey()
#crop_routine()

#fp = "/pers_files/Combined_final/to_crop/train/robotcell_all1_color_2021-04-08-10-11-34.jpg"
#img = cv2.imread(fp)[:515,47:1024-47]
#cv2.imshow("double",img)
#cv2.waitKey()

# front = random.sample(list(data_pairs_2021),1)[0]
# front = random.sample(list(data_pairs_2021),1)[0]
# x = cv2.imread(os.path.join(data_dir,split,front+".jpg"))
# x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
# trans = T.CropTransform(40,190,910,530,orig_h=1024,orig_w=1024)
# resize = T.Resize(512,512)
# bright = T.RandomBrightness(0.85,1.2)
# light = T.RandomLighting(0.85,1.2)
# aug = T.AugmentationList([trans,bright,light])
# input = T.AugInput(x[:,:,[2,1,0]])
# #trans = T.CropTransform(40,190,910,530)
# new_img = aug(input)
# new_img = new_img.apply_image(x)
# new_img = new_img.astype('uint8')
# new_img = cv2.cvtColor(new_img,cv2.COLOR_RGB2BGR)
# cv2.imshow("lul",new_img)
# cv2.waitKey()
#
#
