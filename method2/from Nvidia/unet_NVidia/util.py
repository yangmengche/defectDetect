import matplotlib.image as mpimg

import glob
import os
import numpy as np
import cv2
import json 
import sklearn
from sklearn.cross_validation import train_test_split

from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D,Lambda, Conv2DTranspose, concatenate
from keras.optimizers import Adam
from keras import backend as K
from scipy.ndimage.measurements import label

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 3
IMAGE_SHAPE = (180, 180)
PIXEL_DEPTH = 256
img_rows = 512
img_cols = 512
def load_images_masks(path_to_images, img_type, img_format, resize, ellipse=False):
    
    if not ellipse:
        image_names = [x for x in os.listdir(path_to_images) if x.endswith('.jpg')]
    else:
        image_names = [x for x in os.listdir(path_to_images) if x.endswith(img_type)]

    image_num = len(image_names)
    images_all = np.empty([image_num, resize[0], resize[1], IMAGE_CHANNELS])
    labels_all = np.empty([image_num, resize[0], resize[1], 1])

    #if not ellipse:
        #image_names = [x.replace('xml', img_type) for x in image_names]

    for image_index in range(image_num):
        image_filename = image_names[image_index]
        image = mpimg.imread(os.path.join(path_to_images, image_filename), format=img_format)
        if ellipse:
            mask = get_mask_seg_ellipse(os.path.join(path_to_images, image_filename))
        else:
            mask = get_mask_seg_polygon(os.path.join(path_to_images, image_filename))

        if resize:
            image = cv2.resize(image, (resize[0], resize[1]))
            mask = cv2.resize(mask, (resize[0], resize[1]))
	#print (image.shape,image_filename)
        images_all[image_index] = np.reshape(image, (resize[0], resize[1], IMAGE_CHANNELS))
        labels_all[image_index] = np.reshape(mask, (resize[0], resize[1], 1))

    return images_all, labels_all

def get_mask_seg_polygon(path_to_img, gray=False):
    img = mpimg.imread(path_to_img)
    
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    basename = os.path.basename(path_to_img)
    
    path_to_json = path_to_img.replace('jpg', 'json')
    
    # print(path_to_json)
    
    with open(path_to_json) as json_file:
        json_get = json.load(json_file)
        pts_list = [np.array(pts['points'], np.int32) for pts in json_get['shapes']]
    
    
    # return np.array(pts, np.int32)
    
    mask = np.zeros_like(img)
    
    # rgb_pred = cv2.cvtColor(im_pred, cv2.COLOR_GRAY2RGB)
    # return mask
    mask = cv2.fillPoly(mask, pts_list, (255, 255, 255))
    
    mask[mask > 0] = 1.
    
    return mask
def get_mask_seg_ellipse(path_to_img):
    """
    """

    # get the image

    img = mpimg.imread(path_to_img)
    basename = os.path.basename(path_to_img)

    # filename_index, e.g. filename = 1.png
    # filename_index = 1, for extracting coordinates
    filename_index = int(os.path.splitext(basename)[0]) - 1
    # print(filename_index)

    path_to_coordinates = path_to_img.replace(basename, 'labels.txt')
    coordinates = load_coordinates(path_to_coordinates)

    mask = np.zeros_like(img)
    mask = cv2.ellipse(mask, 
                       (int(coordinates[filename_index]['x']), int(coordinates[filename_index]['y'])),
                       (int(coordinates[filename_index]['major_axis']), int(coordinates[filename_index]['minor_axis'])),
                       (coordinates[filename_index]['angle'] / 4.7) * 270,
                       0, 
                       360, 
                       (255, 255, 255), 
                       -1)

    mask[mask > 0] = 1.

    # print(coordinates[filename_index]['angle'])

    return mask
def load_test_images(path_to_images, img_type, img_format, resize, ellipse=False):
    image_names = [x for x in os.listdir(path_to_images) if x.endswith(img_type)]
    image_num = len(image_names)
    images_all = np.empty([image_num, resize[0], resize[1], IMAGE_CHANNELS])
    # labels_all = np.empty([image_num, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

    for image_index in range(image_num):
        image_filename = image_names[image_index]
        # print(image_filename)
        image = mpimg.imread(os.path.join(path_to_images, image_filename), format=img_format)
        # mask = get_mask_seg_ellipse(os.path.join(path_to_images, image_filename))

        if resize:
            image = cv2.resize(image, (resize[0], resize[1]))

        images_all[image_index] = np.reshape(image, (resize[0], resize[1], IMAGE_CHANNELS))
        #labels_all[image_index] = np.reshape(mask, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

    return images_all
def test():
    print("test")
