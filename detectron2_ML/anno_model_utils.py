import torch
import numpy as np
import cv2
import os.path as path
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2_ML.sample_json import SAMPLE_JSON_DICT
from skimage import measure

from cv2_utils.cv2_utils import *
from data_and_file_utils.file_utils import get_ending_of_file

def output_model_annos(model_dir,data_dir,splits = None,out_dir = None,min_score = 0.5,endings = ["jpg",'jpeg']):
    if out_dir is None:
        out_dir = path.join(data_dir,'pre_annotations')
    cfg = get_cfg()
    if splits is None:
        typical_splits = ['train','val','test']
        splits = [split for split in typical_splits if path.isdir(path.join(data_dir,split))]
        if not splits:
            splits.append("")

#loading model
    cfg.merge_from_file(path.join(model_dir,"cfg.yaml"))
    cfg.MODEL.WEIGHTS =  path.join(model_dir,'best_model.pth')
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.INPUT.MAX_SIZE_TEST = 0
    model = DefaultPredictor(cfg)
    if isinstance(endings,str):
        endings = list(endings)
#finding files and setting up dirs
    for split in splits:
        split_dir,split_out_dir = path.join(data_dir,split),path.join(out_dir,split)
        files = os.listdir(split_dir)
        files = [file for file in  files if get_ending_of_file(file) in endings]
#making output split directories and outputting files
    os.makedirs(out_dir,exist_ok=False)
    for split in splits:
        os.makedirs(split_out_dir,exist_ok=True)
        fps = [path.join(split_dir,name) for name in files]
        for i,file in enumerate(files):
            img = cv2.imread(fps[i])
            out = model(img)['instances'].to('cpu')
            if min_score > 0:
                out = out[out.scores> min_score]
            masks = out.pred_masks
            masks_np = (masks.permute(1, 2, 0).to('cpu').numpy()).astype('uint8')
            masks_ls = [masks_np[:,:,i] for i in range(masks_np.shape[2])]
            masks_ls = [get_largest_component(mask) for mask in masks_ls] #now 255 or 0
            for j,mask in enumerate(masks_ls):
                out_fp = path.join(out_dir,split, file.split(".")[0] +  f"_mask_mask{j}" + ".png")
                cv2.imwrite(out_fp,mask)
            cv2.imwrite(path.join(out_dir,split,file),img)


def mask_to_json(data_dir: str, label: str, out_dir = None,point_skips = 5,img_endings = ['jpg','jpeg'],mask_endings=['png'],empty_threshold: int = 25) -> None:
    '''
    Takes a directory (data_Dir) filled with masks converts them all to kristians favorite json_format point_skips determine precision. larger => less precision
    '''
    data_dict_orig = SAMPLE_JSON_DICT.copy()
    data_dict_orig['shapes'] = []
    picture_front_and_end = []
    mask_names = [name for name in os.listdir(data_dir) if get_ending_of_file(name) in mask_endings]

    for name in os.listdir(data_dir):
        front = None
        for img_ending in img_endings:
            if name.endswith(img_ending):
                end = len(img_ending)
                front = name[:-(1+end)]
                pair = (front,img_ending)
        if front is not None:
            picture_front_and_end.append(pair)
        else:
            ValueError(f"there is no picture, but got a front{front}. This seems like bug in code")
    # np.random.shuffle(pictures)
    # for name in pictures[:10]:
    # #    name = pictures[0]
    #     fp = path.join(data_dir,name)
    #     img = cv2.imread(fp)
    #
    # #    x0,y0,x1,y1 = (900,500,800+910,600+530)
    #
    #     img_crop = tr.apply_image(img)
    # #    cv2.imshow("win",img)
    #     cv2.imshow("wi",img_crop)
    #
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    label_name = 'filet'
    shapes_dict = {'line_color': None, 'fill_color': None, 'label': label, 'shape_type': 'polygon', 'points': None,
                   'flags': {}, 'group_id': None}
#    print(picture_front_and_end)
    for front,ending in picture_front_and_end:
        name = front + "." + ending
        suited_masks = [mask for mask in mask_names if mask.startswith(front)]
        for mask in suited_masks: mask_names.remove(mask)
        fp = path.join(data_dir, name)
        img_orig = cv2.imread(fp)
        masks = [cv2.imread(path.join(data_dir,name),cv2.IMREAD_GRAYSCALE) for name in suited_masks]
        img = img_orig
        data_dict = data_dict_orig.copy()
        data_dict['imagePath'] = name
        data_dict['imageHeight'] = img.shape[0]
        data_dict['imageWidth'] = img.shape[1]
        data_dict['shapes'] = []

        for a_mask in masks:
            ground_truth_binary_mask = a_mask
            fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
            #encoded_ground_truth = encode(fortran_ground_truth_binary_mask)
            # ground_truth_area = mask.area(encoded_ground_truth)
            # ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
            contours = measure.find_contours(ground_truth_binary_mask, .5)
            if a_mask.sum() > empty_threshold:
                segmentations = []
                biggest_contour = np.argmax([len(contour) for contour in contours])
    #            for contour in contours:
                contour = contours[biggest_contour]
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel()
                segmentation = segmentation.reshape(-1, 2)
                segmentation = segmentation.astype(np.float)
                point_ls = segmentation[:-1][::5].tolist()
                new_shape = shapes_dict.copy()
                new_shape['points'] = point_ls
                data_dict['shapes'].append(new_shape)
        cv2.imwrite(path.join(out_dir, name), img)
        with open(path.join(out_dir, front + ".json"), 'w+') as fp:
            json.dump(data_dict, fp)





data_dir = path.join('/pers_files/Combined_final/Filet-10-21/annotated_masks_from_august_18_11')
out_dir = path.join('/pers_files/Combined_final/Filet-10-21/annotations_august_18_11')
os.makedirs(out_dir)
mask_to_json(data_dir,label='filet',out_dir=out_dir)
#data_dir = path.join('/pers_files/Combined_final/Filet-10-21/pre_annotated530x910')
#out_dir = path.join('/pers_files/Combined_final/Filet-10-21/pre_annotated_masks')
#model_dir = path.join('/pers_files/Combined_final/cropped/output/trials/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_48_output')
#output_model_annos(model_dir,data_dir,out_dir=out_dir,min_score=.5)


    #CANCEL THIS


