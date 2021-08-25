import pandas as pd
from filet.Filet_kpt_Predictor import Filet_ModelTester3,Filet_ModelTester_Aug
import os
import cv2
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
import json
import numpy as np
from detectron2_ML.data_utils import get_file_pairs
seg_model_dir ="/pers_files/spoleben/FRPA_annotering/annotations_crop(180,330,820,1450)/output3/trials/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_12_output"
seg_model_fp = os.path.join(seg_model_dir,'best_model.pth')
seg_model_cfg_fp = os.path.join(seg_model_dir,'cfg.yaml')
discriminator_dir = "/pers_files/mask_models_pad_mask_hyper/classi_net_TV_rect_balanced_mcc_score_fixedf4/model86/trained_model"
#internal parameters. Do not touch this.
pic_dim = 1024
p2_crop_size = [[200, pic_dim - 200], [100, pic_dim - 100]]
p2_resize_shape = (693,618)
#internal parameters. Preprocessing to be done by Haiyan/Iman:
# -Crop / resize to 1024/1024.
# - ensure picture is BGR, uint8 (default when you cv2.imread() )
#output is np.array with shape p3_kpts_nr x 2

tester = Filet_ModelTester_Aug(cfg_fp= seg_model_cfg_fp,chk_fp= seg_model_fp,mask_net_chk_fp=os.path.join(discriminator_dir,"best_model.pth"),p3_kpts_nr=9,print_log=True,record_plots=True,device='cuda:0')


#img = cv2.imread("path/to/example/picture.jpg")
#tester.get_key_points(img)


base_dir = "/pers_files/spoleben/FRPA_annotering/annotations_crop(180,330,820,1450)"
split = 'val'
os.listdir(base_dir)
plot_dir = os.path.join(base_dir, "test_plots")
os.makedirs(plot_dir, exist_ok=True)
img_dir = os.path.join(base_dir,split)
#df = pd.read_csv("/pers_files/mask_models_pad_mask19/MSE_net/" +  '/result.csv')
get_file_pairs(img_dir,"",sorted=True)
pairs = {file[:-4] : file for file in os.listdir(img_dir) if file.endswith(".jpg")}
pairs = {key: val for key,val in pairs.items()}

for front, ls in pairs.items():
    print("TREATING",front)
    img_name = front + ".jpg"
    json_name = front + ".json"
    img_fp = os.path.join(img_dir,img_name)
    json_path = os.path.join(img_dir,json_name)
    with open(json_path, "r") as fp:
        gt_dict = json.load(fp)
    save_name = f"{os.path.join(plot_dir,img_name[:-4])}VIZ.jpg"
    if not os.path.isfile(save_name):
        masks = np.zeros((len(gt_dict['shapes']), 1024, 1024))
        for i in range(len(gt_dict['shapes'])):
            masks[i] = polygon2mask((1024, 1024), np.flip(np.array(gt_dict['shapes'][i]['points']), axis=1))
        img = cv2.imread(img_fp)

#    img = img[100:100+1024,600 : 600+1024]
        tester.get_key_points(img,masks)
        fig = plt.figure()
        gs1 = fig.add_gridspec(3,3, hspace=0.001,wspace = 0.001)
        gs1.update(wspace=0.0001, hspace=0.001)  # set the spacing between axes.
        fig,axes = plt.subplots(3,3)
        for i,item in enumerate(tester.plt_img_dict.items()):
           name,plot_img = item
           plot_img = cv2.resize(plot_img,(2048,2048))
           axarr = plt.subplot(gs1[i])
           axarr.imshow(plot_img)
           axarr.set_xticklabels([])
           axarr.set_yticklabels([])
           axarr.set_aspect('equal')
        plt.savefig(save_name,dpi=200)
        plt.close(fig)