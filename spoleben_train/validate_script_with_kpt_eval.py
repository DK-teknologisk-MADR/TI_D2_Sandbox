from filet.mask_discriminator.transformations import PreProcessor_Crop_n_Resize_Box
from filet.test_nms_full_model import kpt_Eval
import os

prepper = PreProcessor_Crop_n_Resize_Box(resize_dims=[255,255], pad=50, mean=[0.2010, 0.1944, 0.2488, 0.0000],std=[0.224, 0.224, 0.224, 1])
p1_model_dir ="/pers_files/spoleben/spoleben_09_2021/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_6_output"
#discriminator_dir = "/pers_files/mask_models_pad_mask_hyper/classi_net_TV_rect_balanced_mcc_score_fixedf4/model86/trained_model"
base_dir = "/pers_files/spoleben/spoleben_09_2021/spoleben_augmented"
split = ''
#internal parameters. Do not touch this.
plot_dir = os.path.join(p1_model_dir,'plot101',split)
os.makedirs(plot_dir,exist_ok=True)
evaluator = kpt_Eval(p1_model_dir,mask_net_chk_fp=None,base_dir=base_dir,split=split,plot_dir=plot_dir,device='cuda:1',mask_encoding='bit_mask',img_size=(450,450),p2_prepper = prepper,p2_object_area_thresh=0.09,skip_phase2=True,p3_kpts_nr=9,kpts_to_plot=[4],supervised = False)
evaluator.evaluate_on_split()




