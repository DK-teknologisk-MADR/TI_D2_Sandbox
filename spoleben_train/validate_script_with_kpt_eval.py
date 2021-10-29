from filet.mask_discriminator.transformations import PreProcessor_Crop_n_Resize_Box
from detectron2_ML.kpt_eval import kpt_Eval
import os

p1_model_dir ="/pers_files/spoleben/spoleben_09_2021/output_11-10/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_158_output"
base_dir = "/pers_files/spoleben/spoleben_09_2021/spoleben_augmented"
split = ''
plot_dir = os.path.join(p1_model_dir,'plot101',split)
os.makedirs(plot_dir,exist_ok=True)
evaluator = kpt_Eval(p1_model_dir,mask_net_chk_fp=None,base_dir=base_dir,split=split,plot_dir=plot_dir,device='cuda:0',mask_encoding='bit_mask',img_size=(450,450),p2_prepper = None,skip_phase2=True,p3_kpts_nr=9,kpts_to_plot=[4],supervised = False,p1_aug_vote_th=4)
evaluator.evaluate_on_split()
