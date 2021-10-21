import json
import yaml
import cv2
import os
import os.path as path
import random
import re
import cv2_utils.cv2_utils
from detectron2_ML.data_utils import get_file_pairs, get_data_dicts
import detectron2.data.transforms as T
import torch
import numpy as np
from copy import deepcopy
from filet.crop_script import unflatten_polygon_ls,coco_polys_to_ls_of_arrays
data_dir = '/home/mads/Projects/filet_10_21'
pictures = [name for name in os.listdir(data_dir) if name.endswith(".jpg") or name.endswith(".jpeg")]
x0, y0, x1, y1 = (800, 300, 800 + 1024, 300 + 1024)
tr = T.CropTransform(x0=x0, y0=y0, w=x1-x0, h=y1-y0)
for name in pictures[:10]:
#    name = pictures[0]
    fp = path.join(data_dir,name)
    img = cv2.imread(fp)

#    x0,y0,x1,y1 = (900,500,800+910,600+530)

    img_crop = tr(img)
    cv2.imshow("win",img)
    cv2.imshow("wi",img_crop)

    cv2.waitKey()
    cv2.destroyAllWindows()
