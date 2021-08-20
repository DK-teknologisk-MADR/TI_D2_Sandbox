import cv2
import numpy as np

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

def centroid_of_mask_in_xy(mask):
    moms = cv2.moments(mask,binaryImage=True)
    cx = moms['m10']/moms['m00']
    cy = moms['m01']/moms['m00']
    return cx,cy

def tensor_pic_to_imshow_np(tens_inp,mask=False):
    tens = tens_inp.to('cpu')
    if tens.ndim == 3: #color pic
        if tens.shape[0] == 1:
            tens_new = tens[1:]
            return tensor_pic_to_imshow_np(tens_new)
        else:
            assert tens.shape[0] == 3, "this is not a 3channel color pic, nor a 1 channel bw pic"
        tens = tens.permute(1,2,0).numpy()
    elif tens.ndim == 2: #color pic
        tens = tens.numpy()
        if mask:
            tens = np.where(tens>0,255,0)
            tens = tens.astype('uint8')
    else:
        raise ValueError("weird picture of shape",tens_inp.shape)
    return tens
