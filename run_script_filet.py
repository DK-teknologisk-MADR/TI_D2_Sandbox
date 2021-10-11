from detectron2_ML.Kpt_Predictor import ModelTester_Aug
import os
import cv2
#CHANGE THE FOLLOWING TO YOUR PATHS
p1_model_dir ="/pers_files/Combined_final/Filet/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_4_output"

#dont touch fixed_parameters
fixed_parameters = { "p2_crop_size" : [[200, 1024 - 200], [100, 1024 - 100]] , "p2_resize_shape" : (693,618)}
kpts_out = 21 #feel free to change this

tester = ModelTester_Aug(os.path.join(p1_model_dir, 'cfg.yaml'), os.path.join(p1_model_dir, 'best_model.pth'), mask_net_chk_fp = None, p3_kpts_nr=kpts_out, p2_crop_size=fixed_parameters["p2_crop_size"], p2_resize_shape = fixed_parameters["p2_resize_shape"], device='cuda:1')
#use method .get_key_points(img) for getting keypoints.
#preprocessing before inserting into model:
#  -crop / resize to 1024/1024.
# - ensure picture is BGR, uint8 (default when you cv2.imread())
# - insert it as numpy array or similar

#OUTPUT is np.array with shape 2 x kpts_out


#UNCOMMENT BELOW AND FILL test_img_path for a sample test
test_img_path = "/pers_files/Combined_final/Filet/val/robotcell_all1_color_2021-02-05-12-58-34.jpg"
img = cv2.imread(test_img_path)





