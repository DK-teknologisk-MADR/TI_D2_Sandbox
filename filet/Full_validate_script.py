import pandas as pd
from filet.Filet_kpt_Predictor import Filet_ModelTester3
import os
import cv2
import matplotlib.pyplot as plt
from detectron2_ML.data_utils import get_file_pairs
seg_model_dir ="/pers_files/Combined_final/Filet/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_4_output"
seg_model_fp = os.path.join(seg_model_dir,'best_model.pth')
seg_model_cfg_fp = os.path.join(seg_model_dir,'cfg.yaml')
discriminator_dir = "/pers_files/mask_models_pad_mask35/classi_net1/model59"


#internal parameters. Do not touch this.
pic_dim = 1024
p2_crop_size = [[200, pic_dim - 200], [100, pic_dim - 100]]
p2_resize_shape = (693,618)
#internal parameters. Preprocessing to be done by Haiyan/Iman:
# -Crop / resize to 1024/1024.
# - ensure picture is BGR, uint8 (default when you cv2.imread() )
#output is np.array with shape p3_kpts_nr x 2

tester = Filet_ModelTester3(seg_model_cfg_fp,seg_model_fp,os.path.join(discriminator_dir,"best_model.pth"),p3_kpts_nr=21,p2_crop_size=p2_crop_size,p2_resize_shape = p2_resize_shape,print_log=False,record_plots=True,device='cuda:1')


#img = cv2.imread("path/to/example/picture.jpg")
#tester.get_key_points(img)


base_dir = "/pers_files/Combined_final/Filet"
split = 'val'
os.listdir(base_dir)
plot_dir = os.path.join(base_dir, "viz_classi_net59")
os.makedirs(plot_dir, exist_ok=True)
img_dir = os.path.join(base_dir,split)
#df = pd.read_csv("/pers_files/mask_models_pad_mask19/MSE_net/" +  '/result.csv')
get_file_pairs(img_dir,"")
pairs = {file[:-4] : file for file in os.listdir(img_dir) if file.endswith(".jpg")}
for front, ls in pairs.items():
    img_name = front + ".jpg"
    img_fp = os.path.join(img_dir,img_name)
    img = cv2.imread(img_fp)
#    img = img[100:100+1024,600 : 600+1024]
    tester.get_key_points(img)
    save_name = f"{os.path.join(plot_dir,img_name[:-4])}VIZ.jpg"
    if not os.path.isfile(save_name):
        fig = plt.figure()
        gs1 = fig.add_gridspec(2,3, hspace=0.001,wspace = 0.001)
        gs1.update(wspace=0.0001, hspace=0.001)  # set the spacing between axes.
        fig,axes = plt.subplots(2,2)
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