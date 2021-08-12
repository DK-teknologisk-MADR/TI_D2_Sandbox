from detectron2_ML.data_utils import get_file_pairs
import numpy as np
import os
import shutil
data_dir = "/pers_files/mask_data_raw"
files = get_file_pairs(data_dir,"")
tup = len(files)*.9,len(files)*.1
ls = [int(x) for x in tup]
splits = np.repeat(["train","val"],ls)
np.random.shuffle(splits)
for fr_file_pair,split in zip(files.items(),splits):
    front,files = fr_file_pair
    for file in files:
        shutil.move(os.path.join(data_dir, "", file ), os.path.join(data_dir, split, file ) )
