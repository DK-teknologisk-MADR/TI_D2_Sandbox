import cv2
import numpy as np
import os
from cv2_utils.cv2_utils import *
from cv2_utils.homography_utils import *
datadir= '/home/mads/homography_test/'
fp_bas = os.path.join(datadir,'withcbbasler.bmp')
fp_kinect = os.path.join(datadir,"withCBkinect.png")
pic_bas = cv2.imread(fp_bas)
pic_bas = cv2.rotate(pic_bas, cv2.ROTATE_180)
pic_kin = cv2.imread(fp_kinect)
checkout_imgs(pic_kin)
basler_jpg_fp = os.path.join(datadir,'basler.jpg')
kinect_jpg_fp = os.path.join(datadir,'kinect.jpg')
cv2.imwrite(basler_jpg_fp,pic_bas)
cv2.imwrite(kinect_jpg_fp,pic_kin)
pic_bas_jpg,polys  = load_img_and_polys_from_front(datadir,'basler')
pic_kin_jpg = cv2.imread(kinect_jpg_fp)
checkout_imgs( [pic_kin_jpg,pic_bas_jpg])
l_to_trim_kinect = int((pic_kin_jpg.shape[1]-pic_kin_jpg.shape[0]*1.6)//2)
pic_kin_jpg=pic_kin_jpg[:,l_to_trim_kinect:(3840-l_to_trim_kinect)]
pic_kin_jpg = cv2.resize(pic_kin_jpg,(pic_bas_jpg.shape[1],pic_bas_jpg.shape[0]))
ret1, pts_from = cv2.findChessboardCorners(pic_bas_jpg, (7,5))
ret2, pts_to = cv2.findChessboardCorners(pic_kin_jpg, (7,5))
Hom = get_homography_from_pts(pts_from,pts_to)
checkboard_warped = warp_perspective_on_pts(pts_from,Hom)
new_polys = [warp_perspective_on_pts(poly,Hom) for poly in polys]
img = put_poly_overlays(pic_kin_jpg,new_polys,colors=[(255,0,0),(0,255,0),(0,0,255),(200,0,200),(200,200,0)])
img = put_poly_overlays(img,[checkboard_warped])

checkout_imgs([pic_bas_jpg,img])
