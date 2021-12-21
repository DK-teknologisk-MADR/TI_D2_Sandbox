from detectron2_ML.Kpt_Predictor import ModelTester_Aug
from detectron2_ML.Kpt_Predictor_Old import ModelTester_Aug as Old
import os
import cv2
from cv2_utils import cv2_utils
#CHANGE THE FOLLOWING TO YOUR PATHS
p1_model_dir ="/pers_files/Combined_final/cropped_output_19-11/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_177_output"

kpts_out = 21 #feel free to change this

tester = ModelTester_Aug(cfg_fp = os.path.join(p1_model_dir, 'cfg.yaml'), chk_fp = os.path.join(p1_model_dir, 'best_model.pth'), img_size=(530,910), device='cuda:0',skip_phase2=True,record_plots=True)
tester2 = Old(cfg_fp = os.path.join(p1_model_dir, 'cfg.yaml'), chk_fp = os.path.join(p1_model_dir, 'best_model.pth'), img_size=(530,910), device='cuda:0',skip_phase2=True,record_plots=True)
#use method tester.get_key_points(img) for getting keypoints.
#preprocessing before inserting into model:
#  -crop / resize to 530 x 910.
# - ensure picture is BGR, uint8 (default when you cv2.imread())
# - insert it as numpy array or similar
#OUTPUT is np.array with shape 2 x kpts_out


#UNCOMMENT BELOW FOR FOR TEST
test_img_path = os.path.join(p1_model_dir,'test_pic_in.jpg')
img_in = cv2.imread(test_img_path)
#pts = tester.get_key_points(img_in)
#test_img_path = os.path.join(p1_model_dir,'test_pic_in.jpg')
#img_in = cv2.imread(test_img_path)
#test_img_expected_path = os.path.join(p1_model_dir,'test_pic_out.jpg')
pts1 = tester.get_key_points(img_in)
pts2 = tester2.get_key_points(img_in)
print(pts1)
print(pts2)
print(pts1-pts2)
img_with_pts1 = cv2_utils.put_circle_overlays(img_in,pts1)
img_with_pts2 = cv2_utils.put_circle_overlays(img_in,pts2)
cv2.imwrite(os.path.join(p1_model_dir,'test_pic_out_test.jpg'),img_with_pts1)
cv2.imwrite(os.path.join(p1_model_dir,'test_pic_out_test2.jpg'),img_with_pts2)
test_img_path = os.path.join(p1_model_dir,'robotcell_all1_color_2021-02-09-10-13-15.jpg')
img_in = cv2.imread(test_img_path)
#pts = tester.get_key_points(img_in)
#test_img_path = os.path.join(p1_model_dir,'test_pic_in.jpg')
#img_in = cv2.imread(test_img_path)
#test_img_expected_path = os.path.join(p1_model_dir,'test_pic_out.jpg')
pts1 = tester.get_key_points(img_in)
pts2 = tester2.get_key_points(img_in)
print(pts1-pts2)
test_img_path = os.path.join(p1_model_dir,'robotcell_all1_color_2021-02-09-10-18-46.jpg')
img_in = cv2.imread(test_img_path)
#pts = tester.get_key_points(img_in)
#test_img_path = os.path.join(p1_model_dir,'test_pic_in.jpg')
#img_in = cv2.imread(test_img_path)
#test_img_expected_path = os.path.join(p1_model_dir,'test_pic_out.jpg')
pts1 = tester.get_key_points(img_in)
pts2 = tester2.get_key_points(img_in)
print(pts1-pts2)

#COMPARE WITH EXPECTED OUTPUT IN MODEL_DIR "test_pic_out"



