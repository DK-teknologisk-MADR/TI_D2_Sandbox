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
    fontColor              = (255,0,0)
    lineType               = 2

    cv2.putText(img,text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)
    return img