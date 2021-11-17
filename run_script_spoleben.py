import numpy as np
from detectron2_ML.Kpt_Predictor import ModelTester_Aug
from cv2_utils import cv2_utils
import os
import cv2
from cv2_utils.cv2_utils import *
#CHANGE THE FOLLOWING TO YOUR PATH
p1_model_dir ="/pers_files/spoleben/spoleben_09_2021/output_27-10/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_130_output"
kpts_out = 7 #Change this to any ODD number

tester = ModelTester_Aug(cfg_fp=os.path.join(p1_model_dir, 'cfg.yaml'), chk_fp = os.path.join(p1_model_dir, 'best_model.pth'), skip_phase2=True, p3_kpts_nr=kpts_out, img_size=(450, 450), device='cuda:1',record_plots=True,p1_aug_vote_th=3,p1_aug_iou_th=0.85,aug_lower_params=(0.85,0.85),aug_upper_params=(1.15,1.15))
#USAGE: use method tester.get_key_points(img) to get keypoints.
#PREPROCESSING before inserting into model:
#  -crop / resize to 450 x 450.
# - ensure picture is BGR, uint8 (default when you cv2.imread())
# - insert it as numpy array or similar

#OUTPUT is np.array with shape 2 x kpts_out on the 450 x 450 picture.


#UNCOMMENT BELOW AND FILL test_img_path FOR TESTING
test_img_path = os.path.join(p1_model_dir,'test_pic_in.jpg')
img_in = cv2.imread(test_img_path)
assert img_in.shape[0] == 450 and img_in.shape[1] == 450 and isinstance(img_in,np.ndarray)
pts = tester.get_key_points(img_in)
print(f"output has class {pts.__class__}, and is of shape {pts.shape}, and with data type{pts.dtype}")   #should return an np array of  2 X kpts_out, of floats.

test_img_expected_path = os.path.join(p1_model_dir,'test_pic_out.jpg')
img_out_exp = cv2.imread(test_img_expected_path)
img_with_pts = cv2_utils.put_circle_overlays(img_in,pts)
img_dict = {img_name : img for img_name,img in zip(['in','out','expected_out'],[img_in,img_with_pts,img_out_exp])}
plot_dict = tester.plt_img_dict
checkout_imgs(plot_dict,'rgb')

