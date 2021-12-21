import pandas as pd
from copy import deepcopy
from torchvision.transforms import ColorJitter,RandomAffine,Normalize,ToTensor,RandomCrop,RandomVerticalFlip,RandomHorizontalFlip,Compose
from cv2_utils.cv2_utils import *
from detectron2.config import get_cfg
import torch
from detectron2_ML.evaluators import Consistency_Evaluator
from detectron2_ML.data_utils import get_data_dicts, register_data , get_file_pairs,sort_by_prefix
from spoleben_train.data_utils import get_data_dicts_masks
splits = ['']
data_dir = '/pers_files/spoleben/spoleben_09_2021/spoleben_not_annotated' #"/pers_files/spoleben/FRPA_annotering/annotations_crop(180,330,820,1450)"
file_pairs = { split : sort_by_prefix(os.path.join(data_dir,split)) for split in splits }
#file_pairs = { split : get_file_pairs(data_dir,split,sorted=True) for split in splits }
COCO_dicts = {split: get_data_dicts_masks(data_dir,split,file_pairs[split],unannotated_ok=True) for split in splits } #converting TI-annotation of pictures to COCO annotations.
data_names = register_data('filet',splits,COCO_dicts,{'thing_classes' : ['spoleben']}) #register data by str name in D2 api
output_dir = f'/pers_files/spoleben/spoleben_09_2021/output_test2'
data = COCO_dicts[""]

#FOR TESTING#
cfg = get_cfg()
cfg.merge_from_file('/pers_files/spoleben/spoleben_09_2021/output_22-10/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_226_output/cfg.yaml')
cfg.OUTPUT_DIR = '/pers_files/spoleben/spoleben_09_2021/output_22-10/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_226_output'
cfg.MODEL.WEIGHTS = '/pers_files/spoleben/spoleben_09_2021/output_22-10/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_226_output/best_model.pth'
cfg.INPUT.MIN_SIZE_TEST=450

data_ls = deepcopy(COCO_dicts[''])
for data in data_ls:
    data['image'] = torch.tensor(cv2.imread(data['file_name'])).permute(2,0,1)
print("PRINTING DATA",data)

#checkout_imgs(tensor_pic_to_imshow_np(data['image']))

pre_augs = Compose([RandomCrop(size=450),RandomVerticalFlip(),RandomHorizontalFlip()])
eval = Consistency_Evaluator(predictor_cfg = cfg,coco_dicts_ls = data_ls,top_n_ious = 10,img_size = (450,450),device = 'cuda:0',min_size_incon = 3000,pre_augs = pre_augs)
eval.process(data_ls)
df = pd.DataFrame(eval.evaluate(),columns=['file_score_dict']).sort_values(by='file_score_dict')
print(df[:10])
print(df[:10].index)

