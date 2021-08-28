import cv2
import numpy as np
import detectron2.data.transforms as T


def get_aug_im(image,augs,poly=None):
    # Define the augmentation input ("image" required, others optional):
    input = T.AugInput(image)
    # Apply the augmentation:
    transform = augs(input)  # type: T.Transform
    image_transformed = input.image  # new image
    #sem_seg_transformed = input.sem_seg  # new semantic segmentation
    result = {'image' : image , 'poly' : poly}

    return result


augs = T.AugmentationList([
    T.RandomBrightness(0.9, 1.1),
    T.RandomFlip(prob=0.5),
    T.RandomCrop("absolute", (640, 640))
])  # type: T.Augmentation
#image = cv2.imread("/home/madsbr/Documents/pics/homography_source_desired_images.jpg",cv2.IMREAD_GRAYSCALE)
#aug_data = get_aug_im(image,augs)
#aug_image = aug_data['image']
#cv2.imshow(aug_image)
#cv2.waitKey()
#cv2.destroyAllWindows()
class CropAndResize(T.Augmentation):
    def __init__(self, scale_range,shift_range=[0,0]):
        self.scale_range = scale_range
        self.shift_range = shift_range

    def __call__(self,aug_input):
        oldx,oldy,oldc = aug_input.image.shape
        scaler = T.RandomExtent(self.scale_range,self.shift_range)(aug_input)
        resizer = T.Resize((oldx,oldy))(aug_input)
        return T.TransformList([scaler,resizer])

