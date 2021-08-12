from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import cv2
from filet.mask_discriminator.model_tester_mask import Model_Tester_Mask
from pytorch_ML.networks import IOU_Discriminator,IOU_Discriminator_01,try_script_model
from detectron2_ML.predictors import ModelTester
import torch
import torchvision
from torchvision.transforms import Normalize
from detectron2.utils.visualizer import Visualizer
from torch.nn import Conv2d
from detectron2.data import MetadataCatalog
from filet.mask_discriminator.transformations import centralize_img_wo_crop
class Filet_ModelTester3(ModelTester):
    ''''
    last updated to C073_D2_docker aug 06.
    '''
    def __init__(self, cfg_fp, chk_fp, mask_net_chk_fp, p3_kpts_nr=21, p2_object_area_thresh = 20000, p2_iou_thresh = 0.92, p2_n_biggest = 3,p2_crop_size = None,p2_resize_shape = None , device ='cuda:0', print_log = False, pad = 19, record_plots = False, ph2_need_sig=False):
        super().__init__(cfg_fp=cfg_fp, chk_fp=chk_fp,device = device)
        assert p2_resize_shape is not None
        assert p2_crop_size is not None
        self.print_log = print_log
        self.kpts_nr = p3_kpts_nr
#        net = IOU_Discriminator(device = device)
        net = IOU_Discriminator_01(device = device)
        net,_ = try_script_model(net,sample_shape=(p2_n_biggest,4,p2_resize_shape[0],p2_resize_shape[1]),device=device)
        self.totensor = torchvision.transforms.ToTensor()
        self.object_area_thresh = p2_object_area_thresh
        self.iou_thresh = 0.9
        self.h,self.w = 1024,1024
        self.p2_crop_size = p2_crop_size
        self.p2_resize_shape = p2_resize_shape
        self.ph2_need_sig=ph2_need_sig
        self.n_biggest = p2_n_biggest # how many should be checked
        if print_log: print("Filet Init :: device is", device)
        tx = [Normalize(mean=[0.485, 0.456, 0.406, 0], std=[0.229, 0.224, 0.225, 1])]
        self.plt_img_dict = []
        self.record_plots = record_plots
        self.model_tester_mask = Model_Tester_Mask(net, mask_net_chk_fp,trs_x=tx,device=device)
        self.device = device
        self.conv = Conv2d(1, 1, (pad, pad), bias=False,padding=(pad//2,pad//2)).requires_grad_(False) #to be put in predictor class
        self.conv = self.conv.to(device)
        self.conv.weight[0] = torch.ones_like(self.conv.weight[0],device=device)
        self.pred_instances = None
    def log(self,str):
        if self.print_log:
            print(str)
    def phase1(self,img):
        '''
        takes img of filets and return predicted tensor of obj_nr x h x w of instance masks
        '''
        if self.print_log: print('starting phase1')
        pred_output = self.predictor(img)
        has_good_score = pred_output['instances'].scores>0.9
        result = pred_output['instances'][has_good_score].pred_masks.float()
        if self.record_plots:
            self.pred_instances = pred_output['instances'].to('cpu')
            self.pred_instances.remove('pred_boxes')

        if self.print_log: print('ending phase1, found ',result.shape[0], " instances " )

        if self.record_plots:
            img_to_plot = img.copy()
            w = Visualizer(img_to_plot, MetadataCatalog.get('filets'), scale=1)
            out2 = w.draw_instance_predictions(self.pred_instances)
            img_plt = out2.get_image()
            self.plt_img_dict["all"] = cv2.cvtColor(img_plt,cv2.COLOR_BGR2RGB)
        return result

    def phase2(self, pred_masks,img):
        '''
        takes tensor obj_nr,h,w and returns
        best_mask, bad_prediction_warning , is_empty
        '''
#        h_low, h_high, w_low, w_high = 128, 896, 0, 1024
        if self.print_log: print("starting phase 2")
        is_empty = False
        bad_warning_flag = False
        if len(pred_masks) == 0:
            is_empty = True
            if self.print_log : print("Phase2: there seems to be objects")
            best_4d = None
        else:
            areas = pred_masks.sum(dim=(1, 2))
            is_small = areas < self.object_area_thresh
            if torch.all( is_small):
                is_empty = True
                best_4d = None
                if self.print_log: print(" phase2 : all objects seems small")
            else:
                biggest_objects = torch.argsort(areas, descending=True)
                self.log("objects ordered after size are" + str( biggest_objects))
                big_masks = pred_masks[biggest_objects[0:self.n_biggest]].unsqueeze(1)
                big_masks_padded = self.conv(big_masks.float())
                big_masks_padded = big_masks_padded.bool().long()
                img_as_batch = img.unsqueeze(0)
                masked_pic = big_masks_padded * img_as_batch
                img4d_full = torch.cat([masked_pic, big_masks], dim=1)

#------------------p2-preprocessing------------------------
                masked_pic_cpu = masked_pic.to('cpu').permute(0,2,3,1).numpy()
                big_masks_cpu = big_masks.to('cpu').permute(0,2,3,1).numpy()
            #    print("got np arrays",masked_pic_cpu.shape,big_masks_cpu.shape)

                masked_pic_cpu_cent = np.zeros_like(masked_pic_cpu)
                big_masks_cpu_cent = np.zeros_like(big_masks_cpu)
                for i in range(masked_pic_cpu.shape[0]):
                    res ,translations , _ = centralize_img_wo_crop(masked_pic_cpu[i])
                    masked_pic_cpu_cent[i] = res
                    t_1,t_2 = translations
                    T = np.float32([[1, 0, t_1], [0, 1, t_2]])
                    res = cv2.warpAffine(big_masks_cpu[i].squeeze(2), T, (self.w, self.h))
                    big_masks_cpu_cent[i] = res[:,:,None]
#                   cv2.imshow("winz",big_masks_cpu_cent[i])
#                    cv2.waitKey()

                masked_pic = torch.tensor(masked_pic_cpu_cent,requires_grad=False,device=self.device).permute(0,3,1,2)
                big_masks = torch.tensor(big_masks_cpu_cent,requires_grad=False,device=self.device).permute(0,3,1,2)
# ------------------p2-preprocessing------------------------
                img4d_to_eval = torch.cat([masked_pic, big_masks], dim=1)
                #print(masked_pic.shape,big_masks.shape)
             #   print("img4d has size", img4d_full.shape)
                img4d_to_eval = img4d_to_eval[:,:,self.p2_crop_size[0][0]:self.p2_crop_size[0][1],self.p2_crop_size[1][0]:self.p2_crop_size[1][1]]
                img4d_resized = torchvision.transforms.functional.resize(img4d_to_eval, self.p2_resize_shape) #obj_nr x 4 x resize_shape
                # pass im4d_resized through model 2
                self.log('PHASE2: resized img to ' + str(self.p2_resize_shape)  + "and inserting to discriminator")
                pred_ious = self.model_tester_mask.get_evaluation(img4d_resized)
                self.log("PHASE2: best instances had iou" + str(pred_ious))
                is_good_mask = pred_ious > self.iou_thresh
             #   print(is_good_mask)

                if torch.any(is_good_mask):
                    ind_to_biggest_objects = torch.argmax(is_good_mask.long())
                    pred_iou = pred_ious[ind_to_biggest_objects]
                    self.log("PHASE2: found good object")
                else:
                    ind_to_biggest_objects = torch.argmax(pred_ious)
                    pred_iou = torch.max(pred_ious)
                    bad_warning_flag = True
                    self.log("PHASE2: found NO good object")
                best_object_index = biggest_objects[ind_to_biggest_objects]
                self.log("PHASE2: CHOOSING " +str( best_object_index))
                best_4d = img4d_full[ind_to_biggest_objects,:, :, :]
                self.log("PHASE2: returning best 4d with shape" + str(best_4d.shape))

                if self.record_plots:

                    img_np = (img.permute(1, 2, 0).to('cpu').numpy() * 255).astype('uint8').copy()
              #      print(img.shape)
                    best_object_index_np = best_object_index.to('cpu').numpy()
              #      print("best index is", best_object_index_np)
                    w = Visualizer(img_np, MetadataCatalog.get('filets'), scale=1)
              #      print(self.pred_instances)
                    if len(self.pred_instances) == 1:
                        instance = self.pred_instances
                    else:
                        picker = np.zeros(len(self.pred_instances)).astype('bool')
                        picker[best_object_index_np] = True
                        instance = self.pred_instances[picker]
               #     print(instance)
                    out2 = w.draw_instance_predictions(instance)
                    img_plt = out2.get_image()
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (50, 50)
                    # fontScale
                    fontScale = 2
                    # Blue color in BGR
                    color = (255, 0,0)
                    # Line thickness of 2 px
                    thickness = 4
                    # Using cv2.putText() method
             #       print(pred_iou.to('cpu').numpy())
                    pred_iou_str = "{:.2f}".format(float(pred_iou.to('cpu').numpy()))
                    string = f"pred_score = {pred_iou_str}"
                    img_plt = cv2.putText(img_plt, string, org, font,
                                        fontScale, color, thickness, cv2.LINE_AA)

                    self.plt_img_dict["chosen"] = img_plt

        return best_4d , is_empty , bad_warning_flag
                # acc_2nd_step()


    def phase3(self, best_4d):
        '''
        takes mask and predicts kpts as cpu numpy object
        '''
        # from best_mask to keypoints
        full_mask = best_4d[3, :, :]
        self.log("full dimension recieved in phase3 is" + str(full_mask.shape))
        is_filet_col = torch.any(full_mask,dim=0).long()
        offset_w_min = is_filet_col.argmax()
        offset_w_max = self.h - 1 - is_filet_col.flip(0).argmax()
        mask = full_mask[:,offset_w_min : offset_w_max]
        h, w = mask.shape
        split_size = (w // self.kpts_nr)
        is_obj_thresh = split_size // 2
        new_w = split_size * self.kpts_nr
        to_crop = w - new_w
        crop_left = to_crop // 2 + 1 if to_crop % 2 else to_crop // 2
        crop_right = to_crop // 2
        new_mask = mask[:,crop_left:w - crop_right]
        split_ls = torch.split(new_mask, split_size, dim=1)
        splits_ts = torch.stack(split_ls).long()
        is_filet = (splits_ts.sum(dim=2) > is_obj_thresh).long()
        bounds = torch.zeros((2, self.kpts_nr), device=self.device)  # upper / lower
        bounds[0, :] = is_filet.argmax(dim=1)
        bounds[1, :] = h - 1 - is_filet.fliplr().argmax(dim=1)
        kpts = np.zeros((2, self.kpts_nr))
        offset_w_min = offset_w_min.to('cpu').numpy()
        kpts[1, :] = bounds.mean(dim=0).to('cpu').numpy()
        kpts[0, :] = offset_w_min + crop_left + np.arange(self.kpts_nr) * split_size + split_size // 2
        self.log( "found kpts" +str( kpts))
        return kpts


    def get_key_points(self,inputs):
        self.plt_img_dict = {}
        img_to_plot = inputs.copy()
        img_to_plot = cv2.cvtColor(img_to_plot, cv2.COLOR_BGR2RGB)
        self.plt_img_dict['raw'] = img_to_plot
        pred_masks = self.phase1(inputs)
        inputs_rgb = cv2.cvtColor(inputs,cv2.COLOR_BGR2RGB) / 255.0

        inputs_ts = torch.tensor(inputs_rgb,device=self.device,requires_grad=False).permute(2,0,1).float()
        #cv2.imshow("window",pred_masks[0].to('cpu').numpy())
        #cv2.waitKey()
        best4d , is_empty, is_warning = self.phase2(pred_masks,inputs_ts)
        if not is_empty:
            kpts = self.phase3(best4d)
        else:
            kpts = np.full((2,self.kpts_nr),-1)

        if self.record_plots:
            output_circle_img = inputs_rgb.copy()
            if not is_empty:
                #    points1 = [300,300]
                #   points2 = [700,700]
                for i, point in enumerate(kpts.transpose()):
                    if i in [6, 14]:
                        color = (0, 0, 255)
                        radius = 10
                    else:
                        color = (255, 0, 0)
                        radius = 5
                    output_circle_img = cv2.circle(output_circle_img, tuple(point.astype('int')), radius, color, 3)
            self.plt_img_dict['circles'] = output_circle_img

        return kpts




#dir_with_everything ="/pers_files/test_model/mask_rcnn_R_50_FPN_3x_4_output/"
#dir_with_everything = "path/to/dir/with/all/the/stuff"
#predictor =Filet_ModelTester2(dir_with_everything + "cfg.yaml",dir_with_everything + "best_model.pth",n_split=21)
#input_img = cv2.imread(dir_with_everything + 'filet.jpg')
#start_time = time.time()
#pred_outputs = predictor(input_img)
#kpts = predictor.get_key_points(input_img)
#print("inference timing :",time.time()-start_time)
#os.listdir("/pers_files/test_model/mask_rcnn_R_50_FPN_3x_4_output")
#print(kpts)
