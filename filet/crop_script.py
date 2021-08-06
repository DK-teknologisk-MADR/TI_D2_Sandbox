import cv2
import os
import random
import re
import detectron2.data.transforms as T
import torch

splits = ['train','val']
data_dir = "/pers_files/Combined_final/Filet"
#data_dir = "/pers_files/test_set"
base_output_dir = f'{data_dir}/output'
model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"

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
data_pairs_2020 , data_pairs_2021 = partition_pairs_by_year(train_pairs)
regex_string = r'2021-04'
double_case_finder = re.compile(regex_string)
split = 'train'
data_pairs_2021 = {key : val for key,val in data_pairs_2021.items() if not double_case_finder.search(key)}
x = cv2.imread(os.path.join(data_dir,split,front+".jpg"))
x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
cv2.imshow("window",x[200:720,50:950])
cv2.waitKey()
_ , data_pairs_val = partition_pairs_by_year(val_pairs)
x = double_case_finder.search('robotcell_all1_color_2021-04-08-13-40-19')


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

class CropAndResize(T.Augmentation):
    def __init__(self, scale_range,shift_range=[0,0]):
        self.scale_range = scale_range
        self.shift_range = shift_range

    def __call__(self,aug_input):
        oldx,oldy,oldc = aug_input.image.shape
        scaler = T.RandomExtent(self.scale_range,self.shift_range)(aug_input)
        resizer = T.Resize((oldx,oldy))(aug_input)
        return T.TransformList([scaler,resizer])



