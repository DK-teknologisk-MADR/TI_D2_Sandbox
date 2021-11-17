import numpy as np
from detectron2_ML.Kpt_Predictor import ModelTester_Aug
from cv2_utils import cv2_utils
import os
import os.path as path
import cv2
from cv2_utils.cv2_utils import *
#CHANGE THE FOLLOWING TO YOUR PATH
p1_model_dir ="/pers_files/spoleben/spoleben_09_2021/output_27-10/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_130_output"
kpts_out = 7 #Change this to any ODD number

output_dir = "/pers_files/to_kristian"
tester = ModelTester_Aug(cfg_fp=os.path.join(p1_model_dir, 'cfg.yaml'), chk_fp = os.path.join(p1_model_dir, 'best_model.pth'), skip_phase2=True, p3_kpts_nr=kpts_out, img_size=(450, 450), device='cuda:1',record_plots=True,p1_aug_vote_th=3,p1_aug_iou_th=0.85,aug_lower_params=(0.85,0.85),aug_upper_params=(1.15,1.15))
#USAGE: use method tester.get_key_points(img) to get keypoints.
#PREPROCESSING before inserting into model:
#  -crop / resize to 450 x 450.
# - ensure picture is BGR, uint8 (default when you cv2.imread())
# - insert it as numpy array or similar

#OUTPUT is np.array with shape 2 x kpts_out on the 450 x 450 picture.


#UNCOMMENT BELOW AND FILL test_img_path FOR TESTING
input_dir = '/pers_files/spoleben/spoleben_09_2021/spoleben_not_annotated_aug'
examples = [
'kinect_20210916_101351_color__ID162aug_nr0.jpg',
'kinect_20210916_101415_color__ID163aug_nr1.jpg',
'kinect_20210916_102427_color__ID190aug_nr1.jpg'
]
current_example = path.join(input_dir,examples[-1])
#test_img_path = os.path.join(p1_model_dir,'test_pic_in.jpg')
img_in = cv2.imread(current_example)
assert img_in.shape[0] == 450 and img_in.shape[1] == 450 and isinstance(img_in,np.ndarray)
pts = tester.get_key_points(img_in)
print(f"output has class {pts.__class__}, and is of shape {pts.shape}, and with data type{pts.dtype}")   #should return an np array of  2 X kpts_out, of floats.

img_with_pts = cv2_utils.put_circle_overlays(img_in,pts)
plot_dict = tester.plt_img_dict
checkout_imgs(plot_dict,'rgb')

for id,name in enumerate(examples):
    current_example = path.join(input_dir, name)
    print(current_example)
    img_in = cv2.imread(current_example)
    assert img_in.shape[0] == 450 and img_in.shape[1] == 450 and isinstance(img_in, np.ndarray)
    pts = tester.get_key_points(img_in)
    plot_dict = tester.plt_img_dict
    output_for_ex = path.join(output_dir,f"example{id}")
    os.makedirs(output_for_ex,exist_ok=True)
    for title,img in plot_dict.items():
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(path.join(output_for_ex,f"{title}_ID{id}.jpg"),img_rgb)


p1_model_dir ="/pers_files/Combined_final/cropped/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_6_output"
input_dir = '/pers_files/Combined_final/cropped/test'

examples = [
'robotcell_all1_color_2021-04-08-13-13-28upper.jpg',
    'robotcell_all1_color_2021-03-26-13-51-05.jpg',
    'robotcell_all1_color_2021-04-08-13-10-17upper.jpg',
]

tester = ModelTester_Aug(cfg_fp=os.path.join(p1_model_dir, 'cfg.yaml'), chk_fp = os.path.join(p1_model_dir, 'best_model.pth'), skip_phase2=True, p3_kpts_nr=kpts_out, img_size=(530, 910), device='cuda:1',record_plots=True,p1_aug_vote_th=3,p1_aug_iou_th=0.93,aug_lower_params=(0.85,0.85),aug_upper_params=(1.15,1.15))
current_example = path.join(input_dir,examples[-1])
#test_img_path = os.path.join(p1_model_dir,'test_pic_in.jpg')
img_in = cv2.imread(current_example)
#assert img_in.shape[0] == 450 and img_in.shape[1] == 450 and isinstance(img_in,np.ndarray)
pts = tester.get_key_points(img_in)
print(f"output has class {pts.__class__}, and is of shape {pts.shape}, and with data type{pts.dtype}")   #should return an np array of  2 X kpts_out, of floats.

img_with_pts = cv2_utils.put_circle_overlays(img_in,pts)
plot_dict = tester.plt_img_dict
checkout_imgs(plot_dict,'rgb')
previous_plot_dict = None
for id,name in enumerate(examples):
    current_example = path.join(input_dir, name)
    print(current_example)
    img_in = cv2.imread(current_example)
    pts = tester.get_key_points(img_in)
    plot_dict = tester.plt_img_dict
    print("UPDATED PLOT DICT",plot_dict != previous_plot_dict)
    output_for_ex = path.join(output_dir,f"example_filet{id}")
    os.makedirs(output_for_ex,exist_ok=True)
    for title,img in plot_dict.items():
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(path.join(output_for_ex,f"{title}_ID{id}.jpg"),img_rgb)
    previous_plot_dict = plot_dict.copy()
