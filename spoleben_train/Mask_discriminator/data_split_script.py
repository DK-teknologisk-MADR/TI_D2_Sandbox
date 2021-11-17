import os
import shutil
import os.path as path
import re
import pandas as pd
data_dict_by_id = {}
data_dir = '/pers_files/spoleben/spoleben_09_2021/spoleben_masks_expert/data_dir'
rex = re.compile('_id(\d+)')
files = os.listdir(data_dir)
for file in files:
    match = rex.search(file)
    id = int(match[1])
    if id in data_dict_by_id:
        data_dict_by_id[id].append(file)
    else:
        data_dict_by_id[id] = [file]

for id,files in data_dict_by_id.items():
    if len(files)<6:
        print(files)


train_dir = path.join(data_dir,'train')
df = pd.read_csv(path.join('/pers_files/spoleben/spoleben_09_2021/spoleben_masks_expert','train','annotering_p2_train.csv'))
is_null = pd.isnull(df).all(axis=1)
df = df[np.logical_not(is_null)]
df.reset_index(inplace=True)
df['index'] = df['index'].astype('int')

val_dir = path.join(data_dir,'val')
df_val = pd.read_csv(path.join('/pers_files/spoleben/spoleben_09_2021/spoleben_masks_expert','val','annotering_p2_val.csv'))
is_null = pd.isnull(df_val).all(axis=1)
df_val = df_val[np.logical_not(is_null)]
df_val.reset_index(inplace=True)
df_val['index']  =  df_val['index'].astype('int')

for id in df['index']:
    for file in data_dict_by_id[id]:
        shutil.copy(path.join(data_dir,file),path.join('/pers_files/spoleben/spoleben_09_2021/spoleben_masks_expert','train',file))
for id in df_val['index']:
    for file in data_dict_by_id[id]:
        shutil.copy(path.join(data_dir,file),path.join('/pers_files/spoleben/spoleben_09_2021/spoleben_masks_expert','val',file))