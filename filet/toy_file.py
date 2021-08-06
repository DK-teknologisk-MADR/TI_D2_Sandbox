import torch
import time
import numpy as np
from torchvision.transforms import ToTensor, Normalize


import cv2
import numpy as np
import skimage as sk
masks_pred = np.zeros((10,10))
masks_pred[6:10,6:10] = 1
masks_pred
#cv2.warpAffine()
coords = np.array(np.nonzero(masks_pred))
cs = coords.mean(axis=1)
np.array([1,0,-cs[0],0,1,-cs[1]])

x = torch.randint(0,2,(1,1024,1024),device='cuda:1')  + 4.0
x.mean()
x.std()
xstart = time.time()
z = torch.nonzero(x).float()
print(z)
cs = z.mean(axis=0)
end = time.time()
print(cs)
print(end-start)
tr = Normalize( mean=[4.5], std=[0.5])
tr(x).mean()
tr(x).std()