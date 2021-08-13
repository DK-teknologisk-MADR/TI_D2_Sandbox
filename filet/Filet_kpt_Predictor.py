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
from torch.utils.data import Dataset,DataLoader
from detectron2.data import MetadataCatalog
from filet.mask_discriminator.transformations import centralize_img_wo_crop,PreProcessor_Box_Crop,PreProcessor


class P2_Dataset(Dataset):
    def __init__(self, prepper):
        self.prepper = prepper
        self.img = None
        self.masks = None

    def set_img_and_masks(self, img, masks):
        self.img = img
        self.masks = masks

    def __getitem__(self, item):
        img4d = self.prepper.preprocess(self.img, self.masks[item])
        return img4d

    def __len__(self):
        return len(self.masks)

class Filet_ModelTester3(ModelTester):
    ''''
    last updated to C073_D2_docker aug 06.
    '''

    def __init__(self, cfg_fp, chk_fp, mask_net_chk_fp, p3_kpts_nr=21, p2_object_area_thresh = 20000, device ='cuda:0', print_log = False, pad = 19, record_plots = False,p2_resize_shape = None,p2_n_biggest = None,p2_crop_size = None):
        super().__init__(cfg_fp=cfg_fp, chk_fp=chk_fp,device = device)
        self.p2_resize_shape=(393,618)
        self.p2_n_biggest = 3
        self.print_log = print_log
        self.kpts_nr = p3_kpts_nr
        self.h,self.w = 1024,1024
        self.iou_thresh = 0.65
#        net = IOU_Discriminator(device = device)
        net = IOU_Discriminator_01(two_layer_head=False,device = device)
        net,_ = try_script_model(net,sample_shape=(self.p2_n_biggest,4,self.p2_resize_shape[0],self.p2_resize_shape[1]),device=device)
        self.totensor = torchvision.transforms.ToTensor()
        self.object_area_thresh = p2_object_area_thresh
        self.p2_preprocessor = PreProcessor([[250,1024-250],[100,1024-100]],resize_dims=(393,618),pad=35,mean=[0,0,0],std=[1,1,1])
        #self.p2_preprocessor = PreProcessor_Box_Crop([[250,1024-250],[100,1024-100]],resize_dims=(393,618),pad=35,mean=[0.2010, 0.1944, 0.2488, 0.0000],std=[0.3040, 0.2964, 0.3694, 1])
        if print_log: print("Filet Init :: device is", device)
        self.plt_img_dict = []
        self.record_plots = record_plots
        self.model_tester_mask = Model_Tester_Mask(net, mask_net_chk_fp,device=device)
        self.device = device
        self.pred_instances = None
        self.p2_dataset = P2_Dataset(self.p2_preprocessor)



    def log(self,str):
        if self.print_log:
            print(str)
    def phase1(self,img):
        '''
        takes img of filets and return predicted tensor of obj_nr x h x w of instance masks
        '''
        if self.print_log: print('starting phase1')
        is_empty = False
        pred_output = self.predictor(img)
        has_good_score = pred_output['instances'].scores>0.9
        result = pred_output['instances'][has_good_score].pred_masks.type(torch.uint8)
        if self.print_log: print('ending phase1, found ',torch.sum(has_good_score), " good instances " )
        if len(result) == 0:
            is_empty = True
        if result.shape[0] >= self.p2_n_biggest:
            sizes = result.sum(axis=(1,2))
            biggest = torch.argsort(sizes)[-3:]
            result = result[biggest]
        if self.print_log: print("phase1 : result has shape", result.shape)
        if self.record_plots:
            self.pred_instances = pred_output['instances'].to('cpu')
            self.pred_instances.remove('pred_boxes')
            img_to_plot = self.img_to_plot2.copy()
            w = Visualizer(img_to_plot, MetadataCatalog.get('filets'), scale=1)
            out2 = w.draw_instance_predictions(self.pred_instances)
            img_plt = out2.get_image()
            self.plt_img_dict["all"] = img_plt
            self.pred_instances = self.pred_instances[has_good_score]
            print("PRED INSTANCES WITH GOOD SCORE",len(self.pred_instances))
            if len(self.pred_instances) >= self.p2_n_biggest:
                self.pred_instances = self.pred_instances[biggest]

                print("NOW LENGTH SHOULD BE 3",len(self.pred_instances))
        return result , is_empty
    def phase2(self, img4d):
        '''
        takes tensor obj_nr,h,w and returns
        best_mask, bad_prediction_warning , is_empty
        '''
#        h_low, h_high, w_low, w_high = 128, 896, 0, 1024
        self.log("sending p2 to evaluation")
        img4d = img4d.to(self.device)
        pred_ious = self.model_tester_mask.get_evaluation(img4d)
        self.log("PHASE2: best instances had score" + str(pred_ious))
        is_good_mask = pred_ious > self.iou_thresh
     #   print(is_good_mask)
        if torch.any(is_good_mask):
            ind_to_biggest_object = torch.argmax(is_good_mask.long())
#            pred_iou = pred_ious[ind_to_biggest_object]
            self.log("PHASE2: found good object")
        else:
            ind_to_biggest_object = torch.argmax(pred_ious)
#            pred_iou = torch.max(pred_ious)
            bad_warning_flag = True
            self.log("PHASE2: found NO good object")
        best_4d = img4d[ind_to_biggest_object,:, :, :]
        self.log("PHASE2: returning best 4d with shape" + str(best_4d.shape))

        if self.record_plots:
            best_object_index_np = ind_to_biggest_object.to('cpu').numpy()
      #      print("best index is", best_object_index_np)
            for i in range(img4d.shape[0]):
                w = Visualizer(self.img_to_plot2, MetadataCatalog.get('filets'), scale=1)
          #      print(self.pred_instances)
                if len(self.pred_instances) == 1:
                    instance = self.pred_instances
                else:
                    picker = np.zeros(len(self.pred_instances)).astype('bool')
                    picker[i] = True
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
                pred_iou_str = "{:.2f}".format(float(pred_ious[i].to('cpu').numpy()))
                string = f"pred_score = {pred_iou_str}"
                img_plt = cv2.putText(img_plt, string, org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)

                self.plt_img_dict[f"chosen{i}"] = img_plt
        return ind_to_biggest_object
                # acc_2nd_step()


    def phase3(self, full_mask):
        '''
        takes mask and predicts kpts as cpu numpy object
        '''
        # from best_mask to keypoints
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
        if self.record_plots:
            self.plt_img_dict = {}
            img_to_plot = inputs.copy()
            img_to_plot = cv2.cvtColor(img_to_plot, cv2.COLOR_BGR2RGB)
            self.img_to_plot2 = img_to_plot.copy()
            self.img_to_plot3 = img_to_plot.copy()
            self.plt_img_dict['raw'] = img_to_plot
        pred_masks, is_empty = self.phase1(inputs)
        pred_masks_np = pred_masks.to('cpu').numpy()

        self.p2_dataset.set_img_and_masks(inputs,pred_masks_np)
        loader = DataLoader(self.p2_dataset,batch_size=self.p2_n_biggest,shuffle=False,pin_memory=True,drop_last=False,num_workers=0)
        img4d = next(iter(loader))
        ind_to_biggest_object = self.phase2(img4d)

        if not is_empty:
            kpts = self.phase3(pred_masks[ind_to_biggest_object])
        else:
            kpts = np.full((2,self.kpts_nr),-1)

        if self.record_plots:
            output_circle_img = self.img_to_plot3.copy()
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
