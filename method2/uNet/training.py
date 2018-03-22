import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
  help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
  help="path to output model")

ap.add_argument("-e", "--epochs", type=int, default=25,
	help="How many epochs, default=25")	
ap.add_argument("-l", "--log",
  help="Export plot data to csv file")
args = vars(ap.parse_args())


import os
import glob
import cv2 as cv
import matplotlib.image as mpimg
import numpy as np
from sklearn.cross_validation import train_test_split
import utils
import iou


img_rows = 512
img_cols = 512



# X, y = loadImage(args['dataset'])

X, y = utils.load_images_masks(args['dataset'], img_type='jpg', img_format='None', resize=(img_rows, img_cols), ellipse=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# cv.imshow('0', X[0])
# for j, row in enumerate(y[0]):
#   for i, col in enumerate(y[0][j]):
#     y[0][j][i] = y[0][j][i] if y[0][j][i] == 0 else 255*256
lbl = np.asarray(y[0])
print(np.unique(lbl))


from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D,Lambda, Conv2DTranspose, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras import backend as K
from scipy.ndimage.measurements import label
import time

def get_small_unet():
  inputs = Input((img_rows, img_cols, 3))
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


# In[38]:


model = get_small_unet()


# In[39]:



# In[51]:

smooth = 1.
model.compile(optimizer=Adam(lr=1e-4), loss=iou.IOU_calc_loss, metrics=[iou.IOU_calc])


# In[52]:

callbacks=[]
if args['log']:
  csv_logger = CSVLogger('{0}.csv'.format(args['log']))
  callbacks.append(csv_logger)


history = model.fit(X_train, y_train, batch_size=32, epochs=args['epochs'], verbose=1, validation_split=0.1, callbacks=callbacks)

model.save(args['model'])
