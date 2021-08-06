import numpy as np
import os

class Tilings():
    '''
    Crop coords are h1,w1 h2,w2
    '''
    def __init__(self,img_size,split_sizes,window_size,crop_size = None):
        self.img_size = img_size
        self.split_sizes = split_sizes
        self.window_size = window_size
        if crop_size is None:
            self.crop_size = 0,0,self.img_size[0],self.img_size[1]
        elif len(crop_size) == 2:
            h1,w1 = crop_size[0]
            h2,w2 = crop_size[1]
            self.crop_size = h1,w1,h2,w2
        elif len(crop_size) == 4:
            self.crop_size = crop_size
        else:
            raise ValueError("cant recognize crop_size. Should be of format (h1,w1,h2,w2)")
        self.anchors = None
        self.build_anchors()

    def build_anchors(self):
        h1,w1,h2,w2 = self.crop_size
        split_h,split_w = self.split_sizes
        win_h,win_w = self.window_size
        splits_h = np.arange(h1,h2-win_h + 1e-09,split_h)
        splits_w = np.arange(w1,w2-win_w+ 1e-09,split_w)
        self.anchors = np.array(np.meshgrid(splits_h,splits_w)).astype('int')
        self.anchors = self.anchors.transpose(1, 2, 0)

    def __getitem__(self, item):
        '''
        Crop coords are h1,w1 h2,w2
        '''
        assert len(item)==2
        assert self.anchors is not None, "build anchors first"
        anchor_h1,anchor_w1 = self.anchors[item]
        anchor_h2,anchor_w2 = anchor_h1 + self.window_size[0] , anchor_w1 + self.window_size[1]
        return (anchor_h1,anchor_w1,anchor_h2, anchor_w2)

    def get_anchor_pts(self):
        n_h,n_w, _ = self.anchors.shape
        return self.anchors.reshape(n_h*n_w,2)

    def get_indices(self):
        n_h,n_w, _ = self.indices.shape
        return self.indices.reshape(n_h*n_w,2)


    def get_crops(self):
        anchors = self.get_anchor_pts()
        res = np.tile(anchors,2)
        res[:,2:] = res[:,2:] + np.array(self.window_size)
        return res

    def crop_img(self,img,crop_coords):
        return img[crop_coords[0]:crop_coords[2],crop_coords[1]:crop_coords[3]]


#data loading:




