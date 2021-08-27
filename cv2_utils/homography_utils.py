import os,cv2
import numpy as np
from cv2_utils.cv2_utils import put_polys

def check_and_mb_squeeze_pts(pts):
    bad_shape = False
    if pts.ndim == 3:
        if pts.shape[1] == 1:
            pts = pts.squeeze(1)
        else:
            bad_shape = True
    elif pts.ndim != 2:
        bad_shape = True
    if bad_shape:
        raise ValueError("pts_from is not right shape. Should be pts x 2 or pts x 1 x 2, but got", pts.shape)
    return pts
def get_homography_from_pts(pts_from,pts_to):
    pts_from = check_and_mb_squeeze_pts(pts_from)
    pts_to = check_and_mb_squeeze_pts(pts_to)
    H, _ = cv2.findHomography(pts_from, pts_to)
    return H

def warp_perspective_on_pts(pts,H):
    pts = check_and_mb_squeeze_pts(pts)
    ones = np.ones(shape=(pts.shape[0],1))
    pts = np.hstack([pts,ones])
    transform = np.matmul(H,pts.transpose()).transpose()
    x = transform[:,:2] / transform[:,2][:,None]
    return x
def warp_polys(polys,H):
    return [warp_perspective_on_pts(poly,H) for poly in polys]

#img = cv2.imread("/home/madsbr/Documents/pics/homography_source_desired_images.jpg",cv2.IMREAD_GRAYSCALE)
#img1 = img[:,:1280//2]
#img2 = img[:,1280//2:]

#cv2.imshow("win",img1)
#cv2.waitKey()
#cv2.destroyAllWindows()
#patternSize = (9,6)
#ret1, corners1 = cv2.findChessboardCorners(img1, patternSize)
#ret2, corners2 = cv2.findChessboardCorners(img2, patternSize)

#cv2.imshow("win",img2)
#cv2.waitKey()
#cv2.destroyAllWindows()
#pts_from = corners1
#pts_to = corners2

#H = get_homography_from_pts(pts_from,pts_to)
#img3 = cv2.warpPerspective(img1,H,dsize=(img1.shape[1],img1.shape[0]))
#cv2.imshow("win2",img3)
#cv2.waitKey()
#cv2.destroyAllWindows()
#new_pts = warp_perspective_on_pts(pts_from,H)
#img4 = put_polys(img=img3,new_polys=[new_pts])

