
import re
from filet_train.compute_IoU import get_ious
import torch
import json
import data_utils
import numpy as np
from predictors import ModelTester
import cv2
from data_utils import get_file_pairs
import os
from numba import prange,njit
from skimage.draw import polygon2mask
import pandas as pd
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from detectron2.structures import Instances
import numpy as np
matplotlib.use('TkAgg')
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,DatasetCatalog
torch.cuda.device(1)
from filet_train.Filet_kpt_Predictor import Filet_ModelTester
print(torch.cuda.current_device())

def get_file_pairs_2021(data_dir,split):
    file_pairs = get_file_pairs(data_dir,split)
    regex_string = r'202[0-9]-'
    year_finder = re.compile(regex_string)
    data_pairs_2021 = {}
    for front, files in file_pairs.items():
        ma = year_finder.search(front)
        if ma:
            if ma.group() == '2020-':
                pass
            elif ma.group() == '2021-':
                data_pairs_2021[front] = files
            else:
                print("warning: unknown matched year")
                print(front, ma.group())

        else:
            print("warning: unknown year")
            print(front)
    return data_pairs_2021

data_dir = '/pers_files/Combined_final/Filet'

base_dir = "/pers_files/Combined_final/Filet/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_4_output"
coco_dicts = data_utils.get_data_dicts(data_dir,'train',file_pairs=get_file_pairs_2021(data_dir,'train'))

data_utils.register_data('filet',['train'],coco_dicts,{'thing_classes' : ['filet']})



cfg_fp = base_dir + "/cfg.yaml"
chk_dir = base_dir + "/best_model.pth"
#fig, axs = plt.subplots(2, 2)

with torch.cuda.device(0):
    predictor = Filet_ModelTester(cfg_fp,chk_dir)
df = pd.DataFrame(columns=['file','ioubig1','ioubig2','n_inst_gt','n_inst_gt'])
file_pairs = get_file_pairs_2021(data_dir,'val')
front,files = file_pairs.popitem()
for file_item in file_pairs.items():
    front,files = file_item
    if files[0].endswith(".jpg"):
        jpg,json_file = files
    else:
        json_file,jpg = files
    img = os.path.join(data_dir,'val',jpg)
    input_img = cv2.imread(img)
    pred_output = predictor(input_img)
    pred_output['instances']=pred_output['instances'][pred_output['instances'].scores>0.6]
    pred_output['instances'].remove('pred_boxes')
    pred_masks = pred_output['instances'].to('cpu').pred_masks.numpy() * 1
    #find biggest objs
    areas = np.zeros(len(pred_masks))
    for i in range(len(pred_masks)):
        areas[i] = cv2.countNonZero(pred_masks[i, :, :])
    ord_area = np.argsort(areas)
    ind_area_max = ord_area[-1]
    if len(ord_area) > 1:
        ind_area_max2 = ord_area[-2]
    else:
        ind_area_max2 = ind_area_max
    output_circle_img= input_img.copy()
    points1 = predictor.get_key_points(input_img)
#    points1 = [300,300]
 #   points2 = [700,700]
    for i, point in enumerate(np.vstack([points1])):
        if i in [6, 14]:
            color = (0, 0, 255)
            radius = 10
        else:
            color = (255, 0, 0)
            radius = 5
        output_circle_img = cv2.circle(output_circle_img, tuple(point.astype('int')), radius, color, 3)
    # print(pred_output.pred_masks)
    v = Visualizer(input_img[:, :, ::-1], MetadataCatalog.get('filets'), scale=1)
    out = v.draw_instance_predictions(pred_output['instances'].to('cpu'))
    z = pred_output['instances'][[ind_area_max]].to('cpu') #also include ind_area_max2?
    w = Visualizer(input_img[:, :, ::-1], MetadataCatalog.get('filets'), scale=1)
    out2 = w.draw_instance_predictions(z)
    filet_choice = np.argmax(areas)
   # plt.figure(2,2)
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.0001, hspace=0.001)  # set the spacing between axes.
    images =[cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB),cv2.cvtColor(out.get_image()[:, :, ::-1],cv2.COLOR_BGR2RGB),
             cv2.cvtColor(out2.get_image()[:, :, ::-1],cv2.COLOR_BGR2RGB),cv2.cvtColor(output_circle_img,cv2.COLOR_BGR2RGB),]
    for i,image in enumerate(images):
        cv2.imwrite(f"/pers_files/pipeline_viz/{front}-phase{i+1}.jpg",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    for i in range(4):
        axarr = plt.subplot(gs1[i])
        axarr.imshow(images[i])
        plt.axis('on')
        axarr.set_xticklabels([])
        axarr.set_yticklabels([])
        axarr.set_aspect('equal')
    plt.show()
    input()



file_pairs = get_file_pairs_2021(data_dir,'train')
front, files = list(file_pairs.items())[0]
iou_dict = {}
#ASSUMES SORTED AFTER KEYS
if files[0].endswith(".jpg"):
    jpg, json_file = files
else:
    json_file, jpg = files
img = os.path.join(data_dir, 'train', jpg)
input_img = cv2.imread(img)

pred_output = predictor(input_img)
print(pred_output)
n_inst_pred = len(pred_output)
ious = np.zeros(n_inst_pred)
# get gt polygons in image:

with open(os.path.join(data_dir,'train',json_file)) as fp:
    gt_dict = json.load(fp)
n_inst_gt = len(gt_dict['shapes'])
masks = np.zeros((n_inst_gt,1024,1024))
for i in range(n_inst_gt):
    masks[i] = polygon2mask((1024,1024),np.flip(np.array(gt_dict['shapes'][i]['points']),axis=1))
masks_pred = pred_output['instances'].pred_masks.to('cpu').numpy().astype('uint8')
get_ious(masks,masks_pred)

v = Visualizer(input_img[:, :, ::-1], MetadataCatalog.get('filets'), scale=1)
out = v.draw_instance_predictions(pred_output['instances'].to('cpu'))
img = cv2.cvtColor(out.get_image()[:, :, ::-1],cv2.COLOR_BGR2RGB)
plt.imshow(img)
get_ious(masks,masks_pred)
cv2.waitKey()
np.flip(masks_pred[0]).shape

contours = cv2.findContours(masks_pred[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
   #     cont_np = np.array([x.flatten() for x in contours[0]])#
pt1,pt2,angle = cv2.fitEllipse(contours[0])
center = (np.array(pt1)+np.array(pt2))/2
img2= img
cnt=contours[0]
rect = cv2.minAreaRect(cnt)
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
mat = cv2.getRotationMatrix2D((512,512),rect[2],1)
mat2 = cv2.warpAffine(img,mat,dsize=(1024,1024))
box = np.int0(cv2.boxPoints(rect))
plt.imshow(cv2.drawContours(img, [box], 0, (36,255,12), 3))
plt.imshow(mat2)



torch.tensor([1,2,3,4,5,6]).reshape(3,2)