#import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import glob
import os
import numpy as np
import cv2
import json
#import pylab  
import sklearn
from sklearn.cross_validation import train_test_split

from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D,Lambda, Conv2DTranspose, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from scipy.ndimage.measurements import label
import time

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

def get_small_unet():
    inputs = Input((img_rows, img_cols,3))
    inputs_norm = Lambda(lambda x: x/127.5 - 1.)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(8, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    
    return model
### IOU or dice coeff calculation
def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)

X, y = load_images_masks('./b/', img_type='jpg', img_format='gray', resize=(512, 512), ellipse=False)
#plt.imshow(X[0,:,:,0])
#pylab.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#print (X_train.shape)
#print (X_test.shape)

model = get_small_unet()
smooth = 1.
model.compile(optimizer=Adam(lr=1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, batch_size=10, epochs=50, verbose=1, validation_split=0.1)
model.save('unet.h5') 

