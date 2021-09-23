import traceback
from detectron2.utils.logger import setup_logger
setup_logger()
from pytorch_ML.compute_IoU import get_ious
import numpy as np
import cv2
import time
from filet.mask_discriminator.model_tester_mask import Model_Tester_Mask
from pytorch_ML.networks import IOU_Discriminator,IOU_Discriminator_01,try_script_model
from detectron2_ML.predictors import ModelTester
import torch
import torchvision
from torchvision.transforms import Normalize
from detectron2.utils.visualizer import Visualizer
from cv2_utils.cv2_utils import get_M_for_mask_balance,warpAffineOnPts
from torch.nn import Conv2d
from torch.utils.data import Dataset,DataLoader
from detectron2.data import MetadataCatalog
from filet.mask_discriminator.transformations import centralize_img_wo_crop,PreProcessor_Box_Crop,PreProcessor,PreProcessor_Crop_n_Resize_Box
import detectron2.data.transforms as Tr
from torchvision.transforms.functional import affine,resize
from cv2_utils.cv2_utils import put_text

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
    '''

    def __init__(self, cfg_fp, chk_fp, mask_net_chk_fp = None, p3_kpts_nr=21, p2_object_area_thresh = None, device ='cuda:0', print_log = False, img_size=(1024,1024), record_plots = False,skip_phase2 = False,kpts_to_plot=None,p2_prepper = None,supervised = False):
        super().__init__(cfg_fp=cfg_fp, chk_fp=chk_fp,device = device)
        self.n_biggest = 4
        self.skip_phase2 = skip_phase2
        self.p2_object_area_thresh = p2_object_area_thresh
        self.print_log = print_log
        self.kpts_nr = p3_kpts_nr
        self.h,self.w = img_size
        print("initializer : img_size is ",img_size )
        self.iou_thresh = 0.7
#        net = IOU_Discriminator(device = device)
        self.totensor = torchvision.transforms.ToTensor()
        self.object_area_thresh = p2_object_area_thresh
        self.supervised = supervised
        #self.p2_preprocessor = PreProcessor([[250,1024-250],[100,1024-100]],resize_dims=(393,618),pad=35,mean=[0,0,0],std=[1,1,1])
        self.p2_preprocessor = p2_prepper
        #self.p2_preprocessor = PreProcessor_Box_Crop([[250,1024-250],[100,1024-100]],resize_dims=(393,618),pad=35,mean=[0.2010, 0.1944, 0.2488, 0.0000],std=[0.3040, 0.2964, 0.3694, 1])
        if print_log: print("Filet Init :: device is", device)
        self.plt_img_dict = {}
        self.record_plots = record_plots
        if self.record_plots is True and kpts_to_plot is None:
            self.kpts_to_plot = [self.kpts_nr//2]
        else:
            self.kpts_to_plot = kpts_to_plot

        if mask_net_chk_fp is not None:
            net = IOU_Discriminator_01(two_layer_head=False, device=device)
            test_img = np.zeros((3, self.h, self.w))
            test_mask = np.zeros((self.h,self.w))
            test_mask[250:254,250:254] = 1
#            net_shape = self.p2_preprocessor.preprocess(np.zeros((3,self.h,self.w)),test_mask)
#            print("NET SHAPE IS",net_shape)
#            net, _ = try_script_model(net, sample_shape=(
#            self.n_biggest, 4, net_shape[1], net_shape[2]), device=device)
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
        if result.shape[0] >= self.n_biggest:
            sizes = result.sum(axis=(1,2))
            biggest = torch.flip(torch.argsort(sizes)[-3:],[0])
            result = result[biggest]
        if self.print_log: print("phase1 : result has shape", result.shape)
        if self.record_plots:
            self.pred_instances = pred_output['instances'].to('cpu')
            self.pred_instances.remove('pred_boxes')
            img_to_plot_raw = self.img_to_plot2.copy()
            if not 'raw' in self.plt_img_dict:
                self.plt_img_dict["raw"] = img_to_plot_raw
            img_to_plot = self.img_to_plot2.copy()
            w = Visualizer(img_to_plot, MetadataCatalog.get('filets'), scale=1)
            out2 = w.draw_instance_predictions(self.pred_instances)
            img_plt = out2.get_image()
            self.plt_img_dict["all"] = img_plt
            self.pred_instances = self.pred_instances[has_good_score]
            print("PRED INSTANCES WITH GOOD SCORE",len(self.pred_instances))
            if len(self.pred_instances) >= self.n_biggest:
                self.pred_instances = self.pred_instances[biggest]
            if result.shape[0] >= self.n_biggest:
                sizes = result.sum(axis=(1, 2))
                biggest = torch.flip(torch.argsort(sizes)[-self.n_biggest:], [0])
                result = result[biggest]
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
        print("PHASE2:following passed",is_good_mask)
     #   print(is_good_mask)
        if torch.any(is_good_mask):
            ind_to_biggest_object = torch.argmin(-is_good_mask.long())
#            pred_iou = pred_ious[ind_to_biggest_object]

            self.log("PHASE2: found good object with index" + str(ind_to_biggest_object))
        else:
            ind_to_biggest_object = torch.argmax(pred_ious)
#            pred_iou = torch.max(pred_ious)
            bad_warning_flag = True
            self.log("PHASE2: found NO good object")

        if self.record_plots:
            pred_ious = pred_ious.cpu().numpy()
            for i in range(len(pred_ious)):
                to_write= self.plt_img_dict[f'best_instances{i+1}']
                string ="p2_sc: {:.2f}".format(float(pred_ious[i]))
                to_write = put_text(to_write,string,pos=(300,300),font_size=2,color=(0,0,255))
                self.plt_img_dict[f'best_instances{i + 1}'] = to_write

        # best_4d = img4d[ind_to_biggest_object,:, :, :]
       # self.log("PHASE2: returning best 4d with shape" + str(best_4d.shape))

      #   if self.record_plots:
      #       best_object_index_np = ind_to_biggest_object.to('cpu').numpy()
      # #      print("best index is", best_object_index_np)
      #       for i in range(img4d.shape[0]):
      #           w = Visualizer(self.img_to_plot2, MetadataCatalog.get('filets'), scale=1)
      #     #      print(self.pred_instances)
      #           if len(self.pred_instances) == 1:
      #               instance = self.pred_instances
      #           else:
      #               picker = np.zeros(len(self.pred_instances)).astype('bool')
      #               picker[i] = True
      #               instance = self.pred_instances[picker]
      #      #     print(instance)
      #           out2 = w.draw_instance_predictions(instance)
      #           img_plt = out2.get_image()
      #           font = cv2.FONT_HERSHEY_SIMPLEX
      #           org = (50, 50)
      #           # fontScale
      #           fontScale = 2
      #           # Blue color in BGR
      #           color = (255, 0,0)
      #           # Line thickness of 2 px
      #           thickness = 4
      #           # Using cv2.putText() method
      #    #       print(pred_iou.to('cpu').numpy())
      #           pred_iou_str = "{:.2f}".format(float(pred_ious[i].to('cpu').numpy()))
      #           string = f"pred_score = {pred_iou_str}"
      #           img_plt = cv2.putText(img_plt, string, org, font,
      #                               fontScale, color, thickness, cv2.LINE_AA)
      #
      #           self.plt_img_dict[f"chosen{i}"] = img_plt
        return ind_to_biggest_object
                # acc_2nd_step()


    def phase3(self, full_mask):
        '''
        takes mask and predicts kpts as cpu numpy object
        '''
        # from best_mask to keypoints
        self.log("Phase3: full dimension recieved in phase3 is" + str(full_mask.shape))
        is_filet_col = torch.any(full_mask,dim=0).long()
        offset_w_min = is_filet_col.argmax()
        offset_w_max = self.w - 1 - is_filet_col.flip(0).argmax()
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
        if self.print_log: print( "found kpts ", str( kpts))
        return kpts


    def get_key_points(self,inputs,true_masks = None):
        if self.print_log: print("recieved image of shape",inputs.shape)
        if self.record_plots:
            self.plt_img_dict = {}
            img_to_plot = inputs.copy()
            img_to_plot = cv2.cvtColor(img_to_plot, cv2.COLOR_BGR2RGB)
            self.img_to_plot2 = img_to_plot.copy()
            self.img_to_plot3 = img_to_plot.copy()
            self.img_to_plot4 = img_to_plot.copy()
            self.plt_img_dict['raw'] = img_to_plot
        if self.print_log: print("starting phsase 1")
        pred_masks, is_empty = self.phase1(inputs,true_masks)
        if not is_empty:
            pred_masks_np = pred_masks.to('cpu').numpy()
            print("NOW SHAPE IS", pred_masks_np.shape)
            if self.skip_phase2:
                print("WARNING: SKIPPING PHASE 2 DUE TO SETTINGS")
                ind_to_biggest_object = 0
            else:
                self.p2_dataset.set_img_and_masks(inputs,pred_masks_np)
                print("NOW SHAPE IS",pred_masks_np.shape)
                loader = DataLoader(self.p2_dataset, batch_size=self.n_biggest, shuffle=False, pin_memory=True, drop_last=False, num_workers=0)
                img4d = next(iter(loader))
                ind_to_biggest_object = self.phase2(img4d)

    #        img_np = img4d[:3,:,:].to('cpu')
    #        print(img_np.shape)
    #        img_np = img_np.permute(0,2,3,1).numpy()
    #        cv2.imshow("AT PHASE 2",img_np[0])
            #cv2.waitKey()
            print("NOW SHAPE IS", pred_masks_np.shape)
            mask, M_rot = self.phase3_preprocess(pred_masks_np[ind_to_biggest_object])
            print("NOW SHAPE IS", pred_masks_np.shape)
            kpts = self.phase3(mask)
            M_rot_inv = cv2.invertAffineTransform(M_rot)
            kpts = self.phase3_postprocess(kpts,M_rot_inv)
        else:
            kpts = np.full((2,self.kpts_nr),-1)

        if self.record_plots and not is_empty:
            output_circle_img = self.img_to_plot3.copy()
            if not is_empty:
                #    points1 = [300,300]
                #   points2 = [700,700]
                for i, point in enumerate(kpts.transpose()):
                    if i in self.kpts_to_plot:
                        color = (0, 0, 255)
                        radius = 10
                    else:
                        color = (255, 0, 0)
                        radius = 5
                    output_circle_img = cv2.circle(output_circle_img, tuple(point.astype('int')), radius, color, 3)
            self.plt_img_dict['circles'] = output_circle_img

        return kpts

    def phase3_preprocess(self,mask_np):
        mask_np = mask_np.astype('uint8')
        print("preprocess1:",mask_np.shape)
        mask, largest_area = self.get_largest_component(mask_np)
        print("preprocess2:",mask.shape)
        M = get_M_for_mask_balance(mask,balance='width')
        mask_new = cv2.warpAffine(mask, M=M, dsize=(mask.shape[1],mask.shape[0]))
        print("preprocess3:",mask_new.shape)
        # cv2.imshow("lol", mask_new)
        if self.print_log: print("Phase3 preprocess : AREA of largest component is",largest_area)
        mask_ts = torch.tensor(mask_new,device=self.device,requires_grad=False)
        print("preprocess3:",mask_ts.shape)

        return mask_ts,M

    def phase3_postprocess(self,kpts,M_inv):
        kpts = warpAffineOnPts(kpts.transpose(),M_inv).transpose()
        return kpts


class Filet_ModelTester_Aug(Filet_ModelTester3):
    def __init__(self, p1_aug_iou_th = 0.92,p1_aug_vote_th=5, **kwargs):
        super().__init__(**kwargs)
        self.p1_aug_iou_th = p1_aug_iou_th
        self.p1_aug_vote_th = p1_aug_vote_th
        t1min = Tr.RandomContrast(intensity_min=0.75, intensity_max=0.85)
        t2min = Tr.RandomSaturation(intensity_min=0.75, intensity_max=0.85)
        t3min = Tr.RandomLighting(0.8)
        t1max = Tr.RandomContrast(intensity_min=1.15, intensity_max=1.25)
        t2max = Tr.RandomSaturation(intensity_min=1.15, intensity_max=1.25)
        t3max = Tr.RandomLighting(1.2)
        self.augsmin = Tr.AugmentationList([t1min, t2min, t3min])
        self.augsmax = Tr.AugmentationList([t1max, t2max, t3max])
        self.aug_nr = 8


    def phase1(self,np_img,gt_masks=None):
        img_trs = []
        img = self.predictor.aug.get_transform(np_img).apply_image(np_img)
        aug_time_start = time.time()
        inp = Tr.AugInput(img)
        img_ts = torch.tensor(img.astype("float32").transpose(2, 0, 1),device=self.device,requires_grad =False)
        img_trs.append({'image': img_ts})
        for i in range(self.aug_nr):
            img_tr = self.augment_img_optics(img,inp,i)
            img_tr = torch.tensor(img_tr.astype("float32").transpose(2, 0, 1),device=self.device,requires_grad=False)
            img_tr = self.augment_img_geometric(img_tr,aug_id=i,inverse=False)
#            if i == 0:
#                img_tr_plot = cv2.cvtColor( 255 * img_tr.to('cpu').numpy().astype('uint8').transpose(1, 2,0),cv2.COLOR_BGR2RGB)
#                cv2.imshow("window",img_tr_plot)
            img_trs.append({'image' : img_tr})
        aug_time = time.time()-aug_time_start
        if self.print_log: print(f"Phase1 : augmented {self.aug_nr } pictures. Batchsize is {len(img_trs)} pictures. aug time: {aug_time}")
        #FOR TESTING
        assert len(img_trs) == self.aug_nr + 1
        with torch.no_grad():
            raw_output= self.predictor.model(img_trs)
        has_good_score = [raw_output[0]['instances'].scores>0.8]
        has_good_score.extend([raw_output[i]['instances'].scores>0.5 for i in range(1,self.aug_nr + 1)])
        masks = [raw_output[i]['instances'][has_good_score[i]].pred_masks for i in range(self.aug_nr + 1)]
        masks_ref = masks[0]
        if self.print_log: print("Phase1 : There are",len(masks_ref),"with good score")

        masks_aug = [masks[i] for i in range(1, self.aug_nr + 1)]
        masks_aug = [self.augment_img_geometric(masks,aug_id=i,inverse=True) for i, masks in enumerate(masks_aug)]
#        img_tr_plot = 255 * masks_aug[0][0].to('cpu').numpy().astype('uint8')
#        cv2.imshow("window2", img_tr_plot)
#        cv2.waitKey()
        sizes = masks_ref.sum(axis=(1, 2))
        self.log("phase1: BEFORE SORTING, sizes are" + str(sizes))
        size_ratio = masks_ref.sum(axis=(1, 2)) / (masks_ref.shape[1] * masks_ref.shape[2])
        print(size_ratio)
        is_big = size_ratio  > self.p2_object_area_thresh
        print(is_big)
        if torch.any(is_big):
            masks_ref = masks_ref[is_big]
            sizes = sizes[is_big]
        elif self.print_log: print("there seems to be no good large filets")
        if masks_ref.shape[0] >= self.n_biggest:
            biggest = torch.argsort(sizes,descending=True)[:self.n_biggest]
        else:
            biggest = torch.argsort(sizes,descending=True)
        masks_ref = masks_ref[biggest]
        if self.print_log: print("Phase1 : There are", len(masks_ref), " masks with large size")
        self.log("phase1: BEFORE SORTING, sizes are" + str(sizes))
        self.log("phase1: The three largest are " + str(biggest))

        vote_nrs = self.get_votes(masks_ref,masks_aug)
        have_enough_votes = vote_nrs >self.p1_aug_vote_th
        best_mask_indices = np.hstack([np.nonzero(have_enough_votes)[0], np.nonzero( np.logical_not(have_enough_votes))[0]])
        if self.print_log: print("best indices are", best_mask_indices)
        #TEST
#        masks_ref_cp = masks_ref.clone()
#        best_mask_indices,vote_nrs,vote_ls = self.get_best_mask_indices(masks_ref,masks_aug)
#        img_plt = masks_ref[0].cpu().numpy().astype('uint8')
#        cv2.imshow("BEFORE best_mask_indices", 255*img_plt)
        masks_ref = masks_ref[best_mask_indices]
#        img_plt2 = masks_ref[0].cpu().numpy().astype('uint8')
#        cv2.imshow("AFTER best_mask_indices", 255*img_plt2)

#        masks_ref_bef_res = masks_ref.clone()
        if self.print_log: print(f" Phase1: deliveringresult of size{masks_ref.shape}")
        if len(best_mask_indices):
            is_empty = False
            masks_ref = resize(img=masks_ref, size=[self.h, self.w])
        else:
            is_empty = True





        if self.record_plots and not is_empty:
            if gt_masks is None and self.supervised:
                raise ValueError("please provide true polys")

            else:
                self.pred_instances = raw_output[0]['instances'].to('cpu')
                self.pred_instances.remove('pred_boxes')
                img_to_plot_proto = img.copy()
                img_to_plot_proto = cv2.cvtColor(img_to_plot_proto,cv2.COLOR_BGR2RGB)
                img_to_plot_dict = {f'img_to_plot{i}' : img_to_plot_proto.copy() for i in range(len(best_mask_indices) + 1)}

                w = Visualizer(img_to_plot_dict['img_to_plot0'], MetadataCatalog.get('filets'), scale=1)
                out2 = w.draw_instance_predictions(self.pred_instances)
                img_plt = out2.get_image()
                img_plt = cv2.resize(img_plt,(self.w,self.h))
                if self.print_log: print("phase1: plotting image of shape", img_plt.shape)

#                cv2.imshow("dinwd",img_plt)
#                cv2.waitKey()
                self.plt_img_dict["all"] = img_plt

                best_instances = raw_output[0]['instances'][has_good_score[0]]
                best_instances.remove('pred_boxes')
                if torch.any(is_big):
                    best_instances = best_instances[is_big]
                best_instances = best_instances[biggest]
                best_instances = best_instances[best_mask_indices]
                best_instances = best_instances.to('cpu')
                best_masks_plot =best_instances.pred_masks
                vote_nrs_reordered = vote_nrs[best_mask_indices]

                #print("TESTING IS PLOT PRED MASK AND PREDMASKCP SAME")
#                print(torch.all(best_instances.pred_masks == masks_ref_cp))
#                print(torch.all(best_instances.pred_masks == masks_ref_bef_res))

#                try:
                if self.supervised:
                    gt_masks = gt_masks.transpose(1,2,0).astype('float')
                    gt_masks = self.predictor.aug.get_transform(gt_masks).apply_image(gt_masks)
                    gt_masks = gt_masks.transpose(2,0,1)
                    ious = get_ious(gt_masks,best_masks_plot.numpy())
                    print(len(ious))
                sizes_pct = 100 * masks_ref.sum(axis=(1, 2)) / (masks_ref.shape[1] * masks_ref.shape[2])
                #                except:
           #     print("failed")
          #      ious = [-1 for _ in range(self.n_biggest)]
                for i in range(1,len(best_mask_indices) + 1):
                    plt_img = img_to_plot_dict[f'img_to_plot{i}']
                    text_string = ""
                    if self.supervised:
                        text_string += "iou: {:.2f}".format(float(ious[i-1]))
                    text_string += f" votes ={vote_nrs_reordered[i-1]}"
                    put_text(plt_img,text_string,pos=(100,100),font_size=2,color=(0,200,75))
                    text_string = "sizes: {:.2f}".format(float(sizes_pct[i-1]))
                    put_text(plt_img,text_string,pos=(100,700),font_size=2,color=(0,200,75))
                    w = Visualizer(plt_img, MetadataCatalog.get('filets'), scale=1)
              #      print("TESTING IS PLOT PRED MASK AND PREDMASKCP SAME")
              #      print(torch.all(best_instances.pred_masks == masks_ref_bef_res.to('cpu')))
                    out2 = w.draw_instance_predictions(best_instances[i-1])
                    img_plt = out2.get_image()
                    img_plt = cv2.resize(img_plt, (self.w, self.h))
                    #                    cv2.imshow("lets plot this",img_plt)
#                    cv2.waitKey()
                    self.plt_img_dict[f"best_instances{i}"] = img_plt
                    if self.supervised:
                        print(f"PHASE1PLOT: got instance{i-1} with iou {ious[i-1]} and votes{vote_nrs_reordered[i-1]}")
                    else:
                        print(f"PHASE1PLOT: got instance{i - 1} with votes{vote_nrs[i - 1]}")

            #     if i == 1:
               #         cv2.imshow("plot_img0",self.plt_img_dict[f"best_instances{1}"])
               #         cv2.waitKey()
        return masks_ref , is_empty

    def get_votes(self,masks_ref,masks_aug):
        masks_aug_ts = torch.cat(masks_aug, dim=0)
        good_masks = self.get_mask_votes(masks_ref, masks_aug_ts, th=self.p1_aug_iou_th)
        vote_ls = self.get_votes_from_good_masks(good_masks)
        vote_nrs = np.array([len(votes) for votes in vote_ls])
        return vote_nrs

    def get_best_mask_indices(self,masks_ref,masks_aug):
        masks_aug_ts = torch.cat(masks_aug, dim=0)
        good_masks = self.get_mask_votes(masks_ref, masks_aug_ts, th=self.p1_aug_iou_th)
        vote_ls = self.get_votes_from_good_masks(good_masks)
        #print([mask.shape[0] for mask in masks_aug])
        #print(vote_ls)
        #self.show_result(masks_ref,masks_aug_ts,vote_ls)
        vote_nrs = np.array([len(votes) for votes in vote_ls])

        best_vote_indices = np.argsort(-vote_nrs) #equialent to argsoting vote_nrs descending
       # print("vote_nrs are",vote_nrs)
        if len(best_vote_indices)>self.n_biggest:
            best_vote_indices = best_vote_indices[:self.n_biggest]
        print("changed best_vote_indices to",best_vote_indices)
        vote_nrs = vote_nrs[best_vote_indices]
        vote_ls = [vote_ls[i] for i in best_vote_indices]
        print("changes vote_nrs according to ",vote_nrs)
        print("changes vote_ls according to ",vote_ls)

        return best_vote_indices, vote_nrs,vote_ls
        #self.show_result(masks_ref, masks_aug, vote_ls)



    def get_largest_component(self, mask):
        '''
        returns:mask of biggest component
        centroids in format (x from left, y from top)
        largest area in pixel count
        discard_flag hints whether its too weird to keep.
        '''
#        discard_flag = False
        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
        # centroids in x from left, y from top.
        largest_components = stats[1:, 4].argsort()[::-1] + 1
        largest_area = stats[largest_components[0], 4]
#        proportions = stats[1:, 4] / largest_area
#        if np.any(np.logical_and(proportions > 0.2, proportions < 1)):
#            discard_flag = True
        mask = np.where(labels == largest_components[0], 1, 0).astype('uint8')
#        centroids = centroids[largest_components[0]]
        return mask, largest_area

    def augment_img_optics(self,img,inp,aug_id):
        if aug_id % (self.aug_nr//2) == 0:
            tr_map = self.augsmin(inp)
        else:
             tr_map =self.augsmax(inp)
        return tr_map.apply_image(img)

    def augment_img_geometric(self,img,aug_id,inverse):
        if aug_id // (self.aug_nr // 4) == 0:
            translate = (61,43)
        elif aug_id // (self.aug_nr // 4) == 1:
            translate =(23,-47)
        elif aug_id // (self.aug_nr // 4) == 2:
            translate =(-26,71)
        else:
            translate = (-35,-51)
        if inverse:
            translate = tuple(-translation for translation in translate)
        args_to_affine = {'angle':0 , 'translate': translate,"scale":1.0,'shear' : 0}
        return affine(img,**args_to_affine)


    def output_to_imshow_format(self,masks, index):
        return 255 * masks[index].to('cpu').numpy().astype('uint8')

    # get_mask_votes(masks_ref,masks_aug,0.9)
    def compute_ious(self,ref_mask, mask_batch, out):
        ref_mask_cps = ref_mask.expand_as(mask_batch)
        num = torch.logical_and(ref_mask_cps, mask_batch).sum(axis=(1, 2))
        den = torch.logical_or(ref_mask_cps, mask_batch).sum(axis=(1, 2))
        if out is None:
            return torch.true_divide(input=num, other=den)
        else:
            torch.true_divide(input=num, other=den, out=out)

    def get_mask_votes(self,masks_ref, masks_aug, th):
        masks_ref_nr = len(masks_ref)
        masks_aug_nr = len(masks_aug)
        ious = [torch.zeros(masks_aug_nr, device=self.device, dtype=torch.float) for i in range(masks_ref_nr)]
        for i in range(masks_ref_nr):
            self.compute_ious(masks_ref[i], masks_aug, out=ious[i])
        good_masks = [iou > th for iou in ious]
        return good_masks

    def get_votes_from_good_masks(self,good_masks):
        votes_ls = [bool_mask.nonzero().flatten() for bool_mask in good_masks]
        return votes_ls

    def show_result(self,masks_ref, masks_aug, vote_ls):
        for ref_mask_id in range(len(masks_ref)):
            voters = vote_ls[ref_mask_id]
            print(voters)
            img_cv = self.output_to_imshow_format(masks_ref, ref_mask_id)
            cv2.imshow(f"mask_orig{ref_mask_id}", img_cv)
            for voter in voters:
                img_cv = self.output_to_imshow_format(masks_aug, voter)
                cv2.imshow(f"mask{voter}", img_cv)
            cv2.waitKey()
            cv2.destroyAllWindows()



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
