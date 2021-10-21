import torch
import numpy as np
import cv2
import os.path as path
data_dir = path.join('C:/','work_projects','model_to_anno_utils','in')
out_dir = path.join('C:/','work_projects','model_to_anno_utils','out')

##TODO:: PAK DET IND I EN FUNKTION AF INDIR OUTDIR
def get_output_masks(img : np.ndarray) -> torch.tensor:
    mids = torch.arange(0,8,3)
    masks = torch.zeros((3,110,110)).bool()
    for i, mid in enumerate(mids):
        masks[i,mid:(mid+10),mid:(mid+10)] = True
    return masks
files = os.listdir(data_dir)
fps = [path.join(data_dir,name) for name in files]
for i,file in enumerate(files):
    img = cv2.imread(fps[i])
    img = cv2.resize(img,dsize=(110,110))
    masks = get_output_masks(img)
    masks_np = (masks.permute(1, 2, 0).to('cpu').numpy() * 255).astype('uint8')
    masks_ls = [masks_np[:,:,i] for i in range(masks_np.shape[2])]
    for j,mask in enumerate(masks_ls):
        out_fp = path.join(out_dir, file.split(".")[0] +  f"_mask_mask{j}" + ".png")
        print(out_fp)
        cv2.imwrite(out_fp,mask)
    cv2.imwrite(path.join(out_dir,file),img)


    #CANCEL THIS


