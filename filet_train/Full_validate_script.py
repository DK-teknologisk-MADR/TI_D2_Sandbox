from filet_train.Filet_kpt_Predictor import Filet_ModelTester3
import os
import cv2
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from data_utils import get_file_pairs
seg_model_dir = "/pers_files/Combined_final/Filet/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_4_output"
seg_model_fp = os.path.join(seg_model_dir,'best_model.pth')
seg_model_cfg_fp = os.path.join(seg_model_dir,'cfg.yaml')
discriminator_dir ="/pers_files/mask_models_pad_mask19/model49/trained"
tester = Filet_ModelTester3(seg_model_cfg_fp,seg_model_fp,os.path.join(discriminator_dir,"best_model.pth"),21,print_log=False,record_plots=False,device='cuda:1')
#base_dir = "/pers_files/Combined_final/Filet"
split = 'val'

#plot_dir = os.path.join(discriminator_dir, "viz")
#os.makedirs(plot_dir, exist_ok=True)
#img_dir = os.path.join(base_dir,split)
#pairs = get_file_pairs(base_dir,'val')

# for front, ls in pairs.items():
#     img_name = front + ".jpg"
#     img_fp = os.path.join(img_dir,img_name)
#     img = cv2.imread(img_fp)
#     tester.get_key_points(img)
#     save_name = f"{os.path.join(plot_dir,img_name[:-4])}VIZ.jpg"
#     fig = plt.figure()
#     gs1 = fig.add_gridspec(2,2, hspace=0.001,wspace = 0.001)
#     gs1.update(wspace=0.0001, hspace=0.001)  # set the spacing between axes.
#     fig,axes = plt.subplots(2,2)
#     for i,item in enumerate(tester.plt_img_dict.items()):
#        name,plot_img = item
#        plot_img = cv2.resize(plot_img,(2048,2048))
#        axarr = plt.subplot(gs1[i])
#        axarr.imshow(plot_img)
#        axarr.set_xticklabels([])
#        axarr.set_yticklabels([])
#        axarr.set_aspect('equal')
#     plt.savefig(save_name,dpi=200)
