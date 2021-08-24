from filet.Filet_kpt_Predictor import Filet_ModelTester3
from filet.Filet_kpt_Predictor import Filet_ModelTester_Aug
from detectron2_ML.data_utils import get_file_pairs
import os
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
from skimage.draw import polygon2mask
#CHANGE THE FOLLOWING TO YOUR PATHS
p1_model_dir ="/pers_files/spoleben/FRPA_annotering/annotations_crop(180,330,820,1450)/output3/trials/COCO-InstanceSegmentation"
p2_model_dir ="/pers_files/mask_models_pad_mask35_TV/classi_net_TV_rect_balanced_mcc_score_fixed/model20"

#dont touch fixed_parameters
fixed_parameters = { "p2_crop_size" : [[200, 1024 - 200], [100, 1024 - 100]] , "p2_resize_shape" : (693,618)}
kpts_out = 21 #feel free to change this

tester = Filet_ModelTester_Aug(cfg_fp= os.path.join(p1_model_dir,'cfg.yaml'),chk_fp = os.path.join(p1_model_dir,'best_model.pth'),mask_net_chk_fp = os.path.join(p2_model_dir,"best_model.pth"),p3_kpts_nr=kpts_out,p2_crop_size=fixed_parameters["p2_crop_size"],p2_resize_shape = fixed_parameters["p2_resize_shape"],device='cuda:0',record_plots=True)
#use method .get_key_points(img) for getting keypoints.
#preprocessing before inserting into model:
#  -crop / resize to 1024/1024.
# - ensure picture is BGR, uint8 (default when you cv2.imread())
# - insert it as numpy array or similar

#OUTPUT is np.array with shape 2 x kpts_out


#UNCOMMENT BELOW AND FILL test_img_path for a sample test




base_dir = "/pers_files/spoleben/FRPA_annotering/annotations_crop(180,330,820,1450)"
split = 'val'
img_dir = os.path.join(base_dir,split)
plot_dir = os.path.join(base_dir,"plot_phase_1_aug")
os.makedirs(plot_dir,exist_ok=True)
#df = pd.read_csv("/pers_files/mask_models_pad_mask19/MSE_net/" +  '/result.csv')
get_file_pairs(img_dir,"")
pairs = {file[:-4] : file for file in os.listdir(img_dir) if file.endswith(".jpg")}
for front, ls in pairs.items():
    img_name = front + ".jpg"
    json_name = front + ".json"
    img_fp = os.path.join(img_dir,img_name)
    json_path = os.path.join(img_dir,json_name)
    img = cv2.imread(img_fp)
#    img = img[100:100+1024,600 : 600+1024]
    save_name = f"{os.path.join(plot_dir,img_name[:-4])}VIZ.jpg"
    with open(json_path,"r") as fp:
        gt_dict = json.load(fp)
    if not os.path.isfile(save_name):
        masks = np.zeros((len(gt_dict['shapes']),1024,1024))
        for i in range(len(gt_dict['shapes'])):
            masks[i] = polygon2mask((1024, 1024), np.flip(np.array(gt_dict['shapes'][i]['points']), axis=1))
        tester.phase1(img,masks)
        fig = plt.figure()
        gs1 = fig.add_gridspec(2, 3, hspace=0.001, wspace=0.001)
        gs1.update(wspace=0.0001, hspace=0.001)  # set the spacing between axes.
        fig, axes = plt.subplots(2, 3)
        for i, item in enumerate(tester.plt_img_dict.items()):
            name, plot_img = item
            plot_img = cv2.resize(plot_img, (2048, 2048))
            axarr = plt.subplot(gs1[i])
            axarr.imshow(plot_img)
            axarr.set_xticklabels([])
            axarr.set_yticklabels([])
            axarr.set_aspect('equal')
        plt.savefig(save_name, dpi=200)
        plt.close(fig)
    tester.plt_img_dict = {}



