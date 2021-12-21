from detectron2_ML.kpt_eval import kpt_Eval
import os

p1_model_dir ="/pers_files/Combined_final/cropped_output_19-11/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_177_output"
#discriminator_dir = "/pers_files/mask_models_pad_mask_hyper/classi_net_TV_rect_balanced_mcc_score_fixedf4/model86/trained_model"
base_dir = "/pers_files/Combined_final/Filet-10-21/annotated_total"
split = 'val'
#internal parameters. Do not touch this.
plot_dir = os.path.join(p1_model_dir,'plot101',"prod_lol"+split)
os.makedirs(plot_dir,exist_ok=True)
#it evaluates modeltester_AUG fro kpt_predictor.py and print out plots
evaluator = kpt_Eval(p1_model_dir,base_dir=base_dir,split=split,plot_dir=plot_dir,device='cuda:1',mask_encoding='poly',img_size=(530,910),p2_object_area_thresh=0.09,skip_phase2=True,p3_kpts_nr=21,kpts_to_plot=[6,14])
evaluator.evaluate_on_split()