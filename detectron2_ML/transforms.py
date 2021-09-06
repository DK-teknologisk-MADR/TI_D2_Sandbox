import cv2
import numpy as np
import detectron2.data.transforms as T
from shapely.geometry import Polygon
from cv2_utils.cv2_utils import *

import logging
import numpy as np
import pycocotools.mask as mask_util
import torch
from PIL import Image

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
def get_aug_im(image,augs,poly=None):
    # Define the augmentation input ("image" required, others optional):
    input = T.AugInput(image)
    # Apply the augmentation:
    transform = augs(input)  # type: T.Transform
    image_transformed = input.image  # new image
    #sem_seg_transformed = input.sem_seg  # new semantic segmentation
    result = {'image' : image , 'poly' : poly}
    return result

#image = cv2.imread("/home/madsbr/Documents/pics/homography_source_desired_images.jpg",cv2.IMREAD_GRAYSCALE)
#aug_data = get_aug_im(image,augs)
#aug_image = aug_data['image']
#cv2.imshow(aug_image)
#cv2.waitKey()
#cv2.destroyAllWindows()

# augs = T.AugmentationList([
#     T.RandomBrightness(0.9, 1.1),
#     T.RandomFlip(prob=0.5),
#     T.RandomCrop("absolute", (640, 640))
# ])



class RemoveSmallest(T.NoOpTransform):
    def __init__(self,min_area_th=None,min_area_pct = None,img_size = None,bybox = True,bySegm = False,):
        super().__init__()
        if min_area_th is not None:
            self.min_area_th = min_area_th
        elif min_area_pct is not None and img_size is not None:
            min_area_th = min_area_pct * img_size[0] * img_size[1]
        else:
            raise ValueError("RemoveSmallest Transform needs either min_area_th, or both min_area_pct and img_size")
        self.byBox = bybox
        self.bySegm = bySegm

    def apply_box(self, coords: np.ndarray):
        if self.byBox:
            area = (coords[:,2]-coords[:,0])*(coords[:,3]-coords[:,1])
            are_too_small = area < self.min_area_th
            coords[are_too_small] = 0
        return coords

    def apply_polygons(self, polygons: list) -> list:
        if self.bySegm:
            are_too_small =[Polygon(poly).area < self.min_area_th for poly in polygons]
            print(are_too_small)
            print(len(polygons))
            polygons = [[] if is_too_small else poly for poly , is_too_small in zip(polygons,are_too_small)]
        return polygons


#tr = RemoveSmallest(min_area_th=5000,min_area_pct=None,bybox=True,byPolygon=True)
#x = tr.apply_polygons(polys)
#[Polygon(poly).area for poly in polys]

# data_dir = "/pers_files/Combined_final/cropped/test"
# front = 'robotcell_all1_color_2020-11-02-12-55-11'
# img,polys = load_img_and_polys_from_front(data_dir,front)
# img_overlay = put_poly_overlays(img,[polys[2]])
# checkout_imgs(img_overlay)
class CropAndRmPartials(T.CropTransform):
    def __init__(self,partial_crop_pct,x0,y0,w,h,orig_w,orig_h):
        super().__init__(x0,y0,w,h,orig_w,orig_h)
        self.partial_crop_pct = partial_crop_pct

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        new_boxes = super().apply_box(box)
        print(box)
        print(new_boxes)
        new_area = (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])
        pcts = new_area / area
        print(pcts)
        are_too_small = pcts<self.partial_crop_pct
        new_boxes[are_too_small] = 0
        return new_boxes

#
# tr = CropAndRmPartials(0.5,220,50,500,300,530,910)
# checkout_imgs(tr.apply_image(img))
# boxes = []
# for poly in polys:
#     min_x,min_y = poly.min(0)
#     max_x,max_y = poly.max(0)
#     box = (min_x,min_y,max_x,max_y)
#     boxes.append(box)
# boxes = np.array(boxes)
# boxes
# tr.apply_box(boxes)


class CropAndResize(T.Augmentation):
    def __init__(self, scale_range,shift_range=[0,0]):
        self.scale_range = scale_range
        self.shift_range = shift_range

    def __call__(self,aug_input):
        oldx,oldy,oldc = aug_input.image.shape
        scaler = T.RandomExtent(self.scale_range,self.shift_range)(aug_input)
        resizer = T.Resize((oldx,oldy))(aug_input)
        return T.TransformList([scaler,resizer])

