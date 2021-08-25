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

class kpt_Eval():
    def __init__(self,p1_model_dir,base_dir,split,plot_dir,device = 'cuda:1',save_plots = True,mask_encoding = 'poly'):
        self.save_plots = save_plots
        self.device = device
        self.p1_model_dir = p1_model_dir
        #p1_model_dir ="/pers_files/Combined_final/Filet/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_4_output"
        self.p2_model_dir ="/pers_files/mask_models_pad_mask35_TV/classi_net_TV_rect_balanced_mcc_score_fixed/model20" #CHANGE THIS TO DUMMY MODEL
        self.tester = Filet_ModelTester_Aug(cfg_fp= os.path.join(self.p1_model_dir,'cfg.yaml'),chk_fp = os.path.join(self.p1_model_dir,'best_model.pth'),device=device,record_plots=True,print_log=True,img_size=(300,300),skip_phase2=True,p3_kpts_nr=7,kpts_to_plot=[3])
        self.base_dir = base_dir
        self.split = split
        self.plot_dir = plot_dir
        self.mask_encoding = mask_encoding
    #dont touch fixed_parameters
    fixed_parameters = { "p2_crop_size" : [[200, 1024 - 200], [100, 1024 - 100]] , "p2_resize_shape" : (693,618)}
    kpts_out = 9
    #use method .get_key_points(img) for getting keypoints.
    #preprocessing before inserting into model:
    #  -crop / resize to 1024/1024.
    # - ensure picture is BGR, uint8 (default when you cv2.imread())
    # - insert it as numpy array or similar

    #OUTPUT is np.array with shape 2 x kpts_out


    #UNCOMMENT BELOW AND FILL test_img_path for a sample test

    def save_plots_from_tester(self,save_name,overwrite_ok = False):
        if not os.path.isfile(save_name) or overwrite_ok:
            fig = plt.figure()
            gs1 = fig.add_gridspec(3, 3, hspace=0.001, wspace=0.001)
            gs1.update(wspace=0.0001, hspace=0.001)  # set the spacing between axes.
            fig, axes = plt.subplots(3, 3)
            for i, item in enumerate(self.tester.plt_img_dict.items()):
                name, plot_img = item
                plot_img = cv2.resize(plot_img, (2048, 2048))
                axarr = plt.subplot(gs1[i])
                axarr.imshow(plot_img)
                axarr.set_xticklabels([])
                axarr.set_yticklabels([])
                axarr.set_aspect('equal')
            plt.savefig(save_name, dpi=200)
            plt.close(fig)
        else:
            print("skipped plotting ",save_name, "to avoid overwriting")


    def eval_on_picture(self,front,file_ls,plot_fp,phase='all'):
        print(file_ls)
        img_fp = next((x for x in file_ls if x.endswith(".jpg")))
        img = cv2.imread(img_fp)
        if self.mask_encoding =='poly':
            json_fp = next((x for x in file_ls if x.endswith(".json")))
            with open(json_fp,"r") as fp:
                gt_dict = json.load(fp)
            masks = np.zeros((len(gt_dict['shapes']),1024,1024))
            for i in range(len(gt_dict['shapes'])):
                masks[i] = polygon2mask((1024, 1024), np.flip(np.array(gt_dict['shapes'][i]['points']), axis=1))
        else:
            assert self.mask_encoding == 'bit_mask'
            np_file = next((x for x in file_ls if x.endswith(".npy")))
            masks = np.load(np_file)
            print("evaluator: found npy mask of shape ", masks.shape)
        if phase == 'phase1':
            self.tester.phase1(img,masks)
        if phase =='all':
            self.tester.get_key_points(img,masks)
        self.save_plots_from_tester(plot_fp)
        self.tester.plt_img_dict = {}


    def evaluate_on_split(self):
        img_dir = os.path.join(self.base_dir,self.split)
        plot_dir = os.path.join(self.base_dir,"plot_phase_1_aug")
        os.makedirs(plot_dir,exist_ok=True)
        #df = pd.read_csv("/pers_files/mask_models_pad_mask19/MSE_net/" +  '/result.csv')
        pairs = get_file_pairs(img_dir,"")
        for front, ls in pairs.items():
            img_name = front + ".jpg"
            json_name = front + ".json"
            file_fps = [ os.path.join(img_dir,name) for name in ls]
            plot_fp = f"{os.path.join(self.plot_dir, img_name[:-4])}VIZ.jpg"
            self.eval_on_picture(front,file_fps,plot_fp=plot_fp,phase='all')



p1_model_dir = '/pers_files/spoleben/FRPA_annotering/annotations_crop(180,330,820,1450)/output3/trials/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_12_output/'
base_dir = '/pers_files/spoleben/FRPA_annotering/annotations_crop(180,330,820,1450)'
split='val'
plot_dir = os.path.join(base_dir,'plots',split)
os.makedirs(plot_dir,exist_ok=True)
evaluator = kpt_Eval(p1_model_dir,base_dir=base_dir,split=split,plot_dir=plot_dir,device='cuda:1',mask_encoding='bit_mask')
evaluator.evaluate_on_split()
