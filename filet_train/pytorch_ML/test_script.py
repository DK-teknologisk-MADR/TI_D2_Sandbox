from filet_train.mask_discriminator.mask_data_loader import rm_dead_data_and_get_ious , get_file_pairs, Filet_Seg_Dataset
from torchvision import transforms

import numpy as np
from filet_train.pytorch_ML.networks import IOU_Discriminator
from Model_Tester_Mask import Model_Tester_Mask
#import scikit_learn.metrics.confusion_matrix
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
data_dir = '/pers_files/mask_data'
train_split = "train"
model_path = '/pers_files/mask_models_pad_mask19'
val_split = "val"

tx = [transforms.Normalize( mean=[0.485, 0.456, 0.406,0.425], std=[0.229, 0.224, 0.225,0.226])]

file_pairs_train = get_file_pairs(data_dir,train_split)
file_pairs_val = get_file_pairs(data_dir,val_split)
data , iou_dict = rm_dead_data_and_get_ious(data_dir,train_split,file_pairs_train)
data_val , iou_dict_val = rm_dead_data_and_get_ious(data_dir,val_split,file_pairs_val)
dt = Filet_Seg_Dataset(data,iou_dict,data_dir,train_split,trs_y_bef=[],mask_only=False)
dt_val = Filet_Seg_Dataset(data_val,iou_dict_val,data_dir,val_split,trs_y_bef=[],mask_only=False)
net = IOU_Discriminator()
x = Model_Tester_Mask(net,f'{model_path}/model49/best_model.pth')
#i = 2
#x0,x1 = 0.8,0.94
#img, iou = dt_val[i]
#if iou < x0:
#    iou = x0
#    cat = 0
#elif iou > x1:
#    iou = x1
#    cat = 2
#else:
#    cat = 1

def get_accs(x0,x1):
    res_ls = []
    for i in range( len( dt_val)):
        img,iou = dt_val[i]

        iou_pred_tr = x.get_evaluation(img.unsqueeze(0))
        iou_pred = iou_pred_tr.to('cpu').numpy()[0][0]
        res_ls.append({'iou':iou , 'iou_pred' : iou_pred})
        print(res_ls[-1])
    df = pd.DataFrame(res_ls)
    df['diff'] = np.abs(df['iou']-df['iou_pred'])
    print(df.mean())
    print(df.std())
    df.to_csv(f"{model_path}/model165/score_dftest")

get_accs(0.8,0.94)

import pandas as pd
df = pd.read_csv(f"{model_path}/model165/score_df")
df['cat_pred']
df["c0"] = np.abs(df['iou_pred']-0.8)
df["c1"] = np.abs(df['iou_pred']-0.87)
df["c2"] = np.abs(df['iou_pred']-0.94)
df["cat_pred"] = np.argmin(df[["c0","c1","c2"]].values,axis=1)
df['cat_pred']
df['cat']
df = df.drop(columns=["c0","c1","c2"])

cm = confusion_matrix(df['cat'].values,df['cat_pred'].values)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2])
disp.plot()
df.groupby(by=["cat"]).mean()
df.groupby(by=["cat"]).std()
