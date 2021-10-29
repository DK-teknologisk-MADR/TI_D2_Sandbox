import numpy as np
import glob
import cv2
import os
import json
import sys

'''
binary categorization of .jpg files. categorization can be 0 or more among the labels given in user_input, based on user input
'''


folder = '/pers_files/spoleben/spoleben_09_2021/spoleben_masks/overview2'
output_folder = '/pers_files/spoleben/spoleben_09_2021/spoleben_masks_annotated/overview/train'
os.makedirs(output_folder,exist_ok=True)
files = glob.glob(folder+'/*.jpg')
files.sort()
user_input = {'m' : 'small' , 'c' : 'covered' , 'b' : 'bad_mask' , 'g': 'good', 's' :'skip'}
if 'n' in user_input: raise ValueError("dont use n as user_input. reserved for next")
for fname in files:
    print(f"annotating {fname}")
    annotations = { label_name : 0 for label_name in user_input.values() }
    basename = os.path.basename(fname)
    name = os.path.splitext(basename)[0]
    img = cv2.imread(fname)
    print("possibilities")
    annotation = None
    done = False
    last_annotation = None
    while not done:
        last_annotation = None
        cv2.imshow(fname, img)
        k = cv2.waitKey(0)
        if k == ord('q'):
            sys.exit('You pressed \'q\' - exiting!')
        if k == ord('n'):
            print(fname + ' - annotations = ' + str(annotations))
            cv2.destroyAllWindows()
            done = True
        for key ,value in user_input.items():
            if k == ord(key):
                last_annotation = value
                annotations[last_annotation] = 1
                print(f"annotated {value}")

        if last_annotation is None and not done:
            print('Oops - did not recognize user input - try again!')
#    output_folder = annotation


    data = {}
    data['imagePath'] = basename
    data['imageHeight'] = img.shape[0]
    data['imageWidth'] = img.shape[1]
    data['shapes'] = []
    for key ,value in user_input.items():
        shape = {}
        shape['label'] = value
        shape['shape_type'] = 'point' # doesn't matter?
        shape['points'] = annotations[value]
        data['shapes'].append(shape)
    print(data)
    print('Saving output:  ' + output_folder +'/ ' +name)
    cv2.imwrite(output_folder +'/ ' +basename, img)
    with open(output_folder + '/' + name + '.json', "w+") as fout:
        json.dump(data, fout, indent=2)
    print(os.path.isfile(output_folder + '/' + name + '.json'))



