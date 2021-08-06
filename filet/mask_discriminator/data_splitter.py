
from filet.mask_discriminator.mask_data_loader import rm_dead_data_and_get_ious
import numpy as np
import os
import shutil
data_dir = "/pers_files/mask_pad_data19_centralized"
files,ious = rm_dead_data_and_get_ious(data_dir,"")
tup = len(files)*.9,len(files)*.1
ls = [int(x) for x in tup]
splits = np.repeat(["train","val"],ls)
np.random.shuffle(splits)
for fr_file_pair,split in zip(files.items(),splits):
    front,files = fr_file_pair
    for file in files:
        shutil.move(os.path.join(data_dir, "", file ), os.path.join(data_dir, split, file ) )
