import json
import os
import cv2
import numpy as np

COLOR_LIST = [(220,120,0),(0,220,120),(155,155,120),(200,185,210),(215,210,142),(215,133,14),(45,170,220),(45,90,132),(0,174,225),(95,0,225),(175,194,44)]
def put_text(img,text,pos = None,font_size = 2,color=(255,0,0)):
    if pos is None:
        pos = (int(img.shape[0]/10,int(img.shape[1]/10)))
# Create a black image
# Write some Text

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = pos
    fontScale              = font_size
    fontColor              = color
    lineType               = 2

    cv2.putText(img,text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)
    return img

def put_polys(img,new_polys,color=(255,0,0)):
    try:
        check = [isinstance(poly,np.ndarray) for poly in new_polys]
    except:
        raise ValueError("new polys should be a list of numpy arrays of size PTR x 2")
    if not all(check):
        raise ValueError("new polys should be a list of numpy arrays of size PTR x 2")
    isClosed = True
    thickness = 2
    new_polys_cv = [poly.astype('int32') for poly in new_polys]
    poly_overlay = cv2.polylines(img.copy(), new_polys_cv,
                          isClosed, color, thickness)
    return poly_overlay

def masks_as_color_imgs(masks):
    return np.repeat(np.expand_dims(masks, -1), 3, axis=-1)


def centroid_of_mask_in_xy(mask):
    if mask.dtype == np.bool:
        mask = mask.copy().astype('uint8')
    moms = cv2.moments(mask,binaryImage=True)
    cx = moms['m10']/moms['m00']
    cy = moms['m01']/moms['m00']
    return cx,cy

def tensor_pic_to_imshow_np(tens_inp,mask=False):
    tens = tens_inp.to('cpu').clone()
    if tens.ndim == 3: #color pic
        if tens.shape[0] == 1:
            tens_new = tens[1:]
            return tensor_pic_to_imshow_np(tens_new)
        else:
            assert tens.shape[0] == 3, "this is not a 3channel color pic, nor a 1 channel bw pic"
        tens = tens.permute(1,2,0).numpy()
        if tens.dtype==np.float32:
            tens *=255
        tens = tens.astype('uint8')
    elif tens.ndim == 2: #color pic
        tens = tens.numpy()
        if mask:
            tens = np.where(tens>0,255,0)
            tens = tens.astype('uint8')
    else:
        raise ValueError("weird picture of shape",tens_inp.shape)
    return tens

def get_M_for_mask_balance(mask,balance='width'):
    #cv2.imshow("orig",mask)
    coords = cv2.findNonZero(mask)
    p1,wh,angle = cv2.minAreaRect(coords)
    if wh[0]<wh[1]:
        angle = angle -90
    center = centroid_of_mask_in_xy(mask)
    M = cv2.getRotationMatrix2D(center=center,angle=angle,scale=1)
    return M
#M = get_M_for_mask_balance(cc)

#mask_new = cv2.warpAffine(cc, M=M, dsize=cc.shape)
#cv2.imshow("lol", mask_new)
#M_inv = cv2.invertAffineTransform(M,mask_new)
#mask_back = cv2.warpAffine(mask_new,M=M_inv,dsize=cc.shape)
#cv2.imshow("lol2", mask_back)

def warpAffineOnPts(pts,M):
    if M.shape[1] == 3:
        ones = np.ones(shape=(len(pts),1))
        pts = np.hstack([pts,ones])

    return M.dot(pts.transpose()).transpose()


def put_circle_overlays(img,pts,colors=COLOR_LIST,alpha=0.9):
        output_circle_img = img.copy()
        #    points1 = [300,300]
        #   points2 = [700,700]
        if isinstance(pts,list):
            pts = np.array(pts)
        if isinstance(pts,np.ndarray):
            if pts.shape[1] != 2:
                if pts.shape[0] == 2:
                    pts = pts.transpose()
                else:
                    raise ValueError("pts should have dim y X 2, but got",pts.shape)
            for i, point in enumerate(pts):
                color = colors[i % len(colors)]
                radius = 10
                output_circle_img = cv2.circle(output_circle_img, tuple(point.astype('int')), radius, color, 3)
            return output_circle_img
        else:
            raise ValueError("pts should be list or nparray, but got smt else")




def put_mask_overlays(img,masks,colors=COLOR_LIST,alpha=0.5):
    if isinstance(masks,np.ndarray):
        if masks.ndim >2:
            masks = [mask.squeeze(0) for mask in np.split(masks,len(masks),axis=0)]
        elif masks.ndim == 2:
            masks = [masks]
    overlay = img.copy()
    if isinstance(colors,tuple):
        colors = [colors]
    color_nr = len(colors)
    for i, mask in enumerate(masks):
        overlay[mask.astype('bool')] = colors[i % color_nr]
    cv2.addWeighted(img, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay



def put_poly_overlays(img,new_polys,colors=COLOR_LIST,alpha = 0.5):
    try:
        check = [isinstance(poly,np.ndarray) for poly in new_polys]
    except:
        raise ValueError("new polys should be a list of numpy arrays of size PTR x 2")
    if not all(check):
        raise ValueError("new polys should be a list of numpy arrays of size PTR x 2")
    if isinstance(colors,tuple):
        colors = [colors]
    new_polys_cv = [poly.astype(np.int32) for poly in new_polys]
    overlay = img.copy()
    color_nr = len(colors)
    print(color_nr)
    for i,poly in enumerate(new_polys_cv):
        filled = cv2.fillPoly(overlay.copy(), [poly],color=colors[i % color_nr])
        cv2.addWeighted(overlay, alpha, filled, 1 - alpha, 0, overlay)
    return overlay
#colors=[(220,120,0),(0,220,120),(155,155,120)]


def checkout_imgs(imgs,channel_mode : str = "") -> None:
    #SET DEFAULTS:
    channel_mode = channel_mode.upper()
    if isinstance(imgs,np.ndarray):
        if imgs.ndim >3:
            raise ValueError("provide either img of shape hxwx3 or hxw, or provide a list of such imgs. shape of input img is",imgs.shape)
        else:
            imgs = [imgs]
    if isinstance(imgs,dict):
        titles = list(imgs.keys())
        imgs = list(imgs.values())
    else:
        titles = [f'img_{i}' for i in range(len(imgs))]
    for title,img in zip(titles,imgs):
        if channel_mode == "RGB" and img.ndim == 3:
            plot_img = img[:,:,::-1]
        else:
            plot_img = img
        cv2.imshow(title,plot_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def load_img_and_polys_from_front(datadir,front):
    json_path = os.path.join(datadir,front + ".json")
    img_path = os.path.join(datadir,front + ".jpg")
    with open(json_path) as fp:
        dict = json.load(fp)
        polys = [np.array(dict['shapes'][i]['points']) for i in range(len(dict['shapes']))]
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("There seems to be no picture at path",img_path)
    return img,polys


def get_largest_component(mask,connectivity = 4):
    '''
    return 1 / 0 mask of largest component
    '''
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    area = stats[:,4]
    area_index_sorted = 1 + np.flip(np.argsort(area[1:])) #area[1: to discard background
    label_of_biggest = area_index_sorted[0]
    return np.where(labels == label_of_biggest,255,0).astype('uint8')

def imread_as_rgb(img,**args):
    img = cv2.imread(img,**args)
    if img is None:
        raise ValueError(f"did not read anything from {img}")
    img = img[:,:,::-1].copy()
    return img