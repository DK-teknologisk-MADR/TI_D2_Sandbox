import numpy as np
import torch
import os.path as path
from detectron2_ML.Kpt_Predictor import ModelTester_Aug
from detectron2.utils.visualizer import Visualizer
from cv2_utils import cv2_utils
from torchvision.transforms import ToTensor
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
import os
import cv2
#CHANGE THE FOLLOWING TO YOUR PATH
p1_model_dir ="/pers_files/spoleben/spoleben_09_2021/output_22-10/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_226_output"
output_dir = p1_model_dir
kpts_out = 7 #Change this to any ODD number
input_fp = '/pers_files/spoleben/spoleben_09_2021/spoleben_not_annotated/kinect_20210916_101916_color__ID176.jpg'
img_full = cv2.imread(input_fp)
#tr = ToTensor()
x0,y0 = 200,100
w,h = 450,450
img_crop = img_full[y0:y0+w,x0:x0+h]
cv2_utils.checkout_imgs(img_crop)
tester = ModelTester_Aug(cfg_fp=os.path.join(p1_model_dir, 'cfg.yaml'), chk_fp = os.path.join(p1_model_dir, 'best_model.pth'), skip_phase2=True, p3_kpts_nr=kpts_out, img_size=(450, 450), device='cuda:1',record_plots=True)
#img_rgb = cv2.cvtColor(img_crop,cv2.COLOR_BGR2RGB)
a_img = torch.as_tensor(img_crop,dtype=torch.float32).permute(2,0,1)
inp = [{'image' : a_img}]
viz = Visualizer(img_crop)

with torch.no_grad():
    out  = tester.predictor.model(inp)[0]['instances']
for pred_category in ['pred_boxes','pred_classes','scores']:
    out.remove(pred_category)
output = viz.draw_instance_predictions(out.to('cpu')).get_image()
cv2_utils.checkout_imgs(output)
tester.get_key_points(img_crop)
names = ['full_image','tiling','segmentation','best_instance','keypoints']
imgs = [img_crop,img_crop,output,tester.plt_img_dict['best_instances1'][:,:,::-1],tester.plt_img_dict['circles'][:,:,::-1]]
for name,img in zip(names,imgs):
    cv2.imwrite(path.join(p1_model_dir,name + ".jpg"),img)
img_dict_rgb = {key : cv2.cvtColor(img,cv2.COLOR_BGR2RGB) for key,img in tester.plt_img_dict.items()}
cv2_utils.checkout_imgs(img_dict_rgb)

#USAGE: use method tester.get_key_points(img) to get keypoints.
#PREPROCESSING before inserting into model:
#  -crop / resize to 450 x 450.
# - ensure picture is BGR, uint8 (default when you cv2.imread())
# - insert it as numpy array or similar

#OUTPUT is np.array with shape 2 x kpts_out on the 450 x 450 picture.


#UNCOMMENT BELOW AND FILL test_img_path FOR TESTING
test_img_path = os.path.join(p1_model_dir,'test_pic_in.jpg')
img_in = cv2.imread(test_img_path)
#
assert img_in.shape[0] == 450 and img_in.shape[1] == 450 and isinstance(img_in,np.ndarray)
pts = tester.get_key_points(img_in)
print(f"output has class {pts.__class__}, and is of shape {pts.shape}, and with data type{pts.dtype}")   #should return an np array of  2 X kpts_out, of floats.

test_img_expected_path = os.path.join(p1_model_dir,'test_pic_out.jpg')
img_out_exp = cv2.imread(test_img_expected_path)
img_with_pts = cv2_utils.put_circle_overlays(img_in,pts)
img_dict = {img_name : img for img_name,img in zip(['in','out','expected_out'],[img_in,img_with_pts,img_out_exp])}
cv2_utils.checkout_imgs(img_dict)

