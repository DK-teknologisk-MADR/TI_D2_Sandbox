import numpy as np
import numba
import cv2
def centralize_img_wo_crop(img,fixed_translation = None):
    height,width,ch = img.shape
    non_zero_coords = np.array(np.nonzero(img))
    flag = False

    if fixed_translation is None:
        center = non_zero_coords.mean(axis=1)
        h_min,w_min = non_zero_coords.min(axis=1)[:2]
        h_max,w_max = non_zero_coords.max(axis=1)[:2]
        t_1 = width // 2-center[1]
        t_2 = height // 2-center[0]
        if t_1 + w_min<0:
            t_1 = 0
            flag = True
        elif t_1+w_max>width:
            t_1 = width-w_max
            flag = True

        if t_2 + h_min < 0:
            t_2 = 0
            flag = True

        elif t_2 + h_max > height:
            t_2 = height - h_max
            flag = True
    else:
        t_1,t_2 = fixed_translation

    T = np.float32([[1, 0, t_1], [0, 1, t_2]])
    img_translation = cv2.warpAffine(img, T, (width, height))
    return img_translation,(t_1,t_2),flag
