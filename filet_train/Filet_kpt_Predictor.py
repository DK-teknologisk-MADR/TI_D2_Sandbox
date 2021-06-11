#colab needs to dl detectron2 from scratch, this only needs to be done once when building docker image ofc.
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
import time
from predictors import ModelTester
import torch
class Filet_ModelTester(ModelTester):
    def __init__(self, cfg_fp, chk_fp, n_split=21, thresh_up=40, thresh_below=10):
        super().__init__(cfg_fp=cfg_fp, chk_fp=chk_fp)
        self.thresh_below = thresh_below
        self.thresh_up = thresh_up
        self.n_split = n_split

    def interpolate_if_needed(self,indices,to_add,x1,x2,x2_ind):
        taxi_norm = np.sum(np.abs(x2 - x1))
        if taxi_norm > self.thresh_up:
            nr_points = np.int(taxi_norm / self.thresh_below)
            to_add.extend(list(np.linspace(x2, x1, nr_points)))
            indices.extend([x2_ind for _ in range(nr_points)])


    def post_process(self,pred_outputs):
        '''
        OBS: NOT final.
        '''
        choice = 0 #SHOULD BE CHOSEN BY SOME PROCESS
        pred_masks = pred_outputs['instances'].pred_masks.to('cpu').numpy()[choice]
        pred_masks = (pred_masks*1).astype('uint8')
        contours = cv2.findContours(pred_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cont_np = np.array([x.flatten() for x in contours[0]])
        boundaries =np.array([ np.min(cont_np), np.max(cont_np) ])
        filet_length = boundaries[1] - boundaries[0]
        to_add = []
        indices = []
        i = 1
        # here we mark what to add
        #plt.scatter(cont_np[:,0], cont_np[:,1], c=list(range(cont_np.shape[0])))
        while (i < cont_np.shape[0]):
            self.interpolate_if_needed(indices,to_add,cont_np[i-1],cont_np[i],i)
            i += 1
        last_ind = len(cont_np) - 1
        self.interpolate_if_needed(indices,to_add,cont_np[last_ind], cont_np[0],last_ind)
        if indices:
          z = np.insert(cont_np, indices, to_add, axis=0)
        else:
          z=cont_np
        ords = z[:, 0].argsort()
        z = z[ords, :]
        split_pts = np.linspace(z[0, 0], z[-1, 0], self.n_split + 1)
        start_window_time =time.time()
        window_start_index = 0
        mean_vals = []
        for window_end_x in split_pts[1:]:
            window_end_index = np.where(z[:, 0] < window_end_x)[0][-1]
            y_vals = z[window_start_index:window_end_index + 1, 1]
            y_bar = np.mean(y_vals)
            mean_over = np.mean(y_vals[y_vals > y_bar])
            mean_under = np.mean(y_vals[y_vals < y_bar])
            mean_vals.append((mean_over + mean_under) / 2)
            window_start_index = window_end_index
        x_vals = split_pts[1:] - 1 / (2 * (split_pts[1:]-split_pts[:-1]))
        #plt.scatter(z[:, 0], z[:, 1], c=list(range(z.shape[0])))
        #plt.scatter(x_vals, mean_vals)
        return np.transpose(np.array([x_vals,mean_vals]))



class Filet_ModelTester2(ModelTester):
    def __init__(self, cfg_fp, chk_fp, n_split=21, thresh_up=40, thresh_below=10):
        super().__init__(cfg_fp=cfg_fp, chk_fp=chk_fp)
        self.thresh_below = thresh_below
        self.thresh_up = thresh_up
        self.n_split = n_split

    def interpolate_if_needed(self,indices,to_add,x1,x2,x2_ind):
        taxi_norm = np.sum(np.abs(x2 - x1))
        if taxi_norm > self.thresh_up:
            nr_points = np.int(taxi_norm / self.thresh_below)
            to_add.extend(list(np.linspace(x2, x1, nr_points)))
            indices.extend([x2_ind for _ in range(nr_points)])


    def post_process(self,pred_outputs):
        '''
        OBS: NOT final.
        '''
        choice = 0 #SHOULD BE CHOSEN BY SOME PROCESS
        pred_mask = pred_outputs['instances'][choice].pred_masks[0]
        print(pred_mask.shape)
        pred_mask.sum(axis=0)
        dims = pred_mask.shape
        nonzeroes = torch.nonzero(pred_mask)
        intervaly = torch.tensor([torch.min(nonzeroes[:, 1]), torch.max(nonzeroes[:, 1]) + 1])
        intervalx = torch.tensor([torch.min(nonzeroes[:, 0]), torch.max(nonzeroes[:, 0]) + 1])
        print(intervalx,intervaly)
        xs = torch.linspace(intervaly[0],intervaly[1],self.n_split,device='cuda')
        intervaly[1] = intervaly[0] + self.n_split * ((intervaly[1] - intervaly[0]) // self.n_split)
        intervalx[1] = intervalx[0] + self.n_split * ((intervalx[1] - intervalx[0]) // self.n_split)
        ys = pred_mask[intervalx[0]:intervalx[1], intervaly[0]:intervaly[1]].sum(axis=0).reshape(self.n_split,
                                                                                                -1).float().mean(axis=1)
        ys = ys + intervaly[0]
        print(ys)
        return torch.vstack([xs,ys]).to('cpu').long().numpy().transpose()

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