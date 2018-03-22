import numpy as np
import glob
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pylab 

import sys
sys.path.append("./")
import util

def predict_evaluation(pred, image, label,i):
    '''
    '''
    # transform gray image to rgb
    img = np.array(image, np.uint8)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # scale pred and mask's pixel range to 0~255
    im_label = np.array(255*label, dtype=np.uint8)
    im_pred = np.array(255*pred, dtype=np.uint8)
    
    # transform both of them to rgb
    rgb_label = cv2.cvtColor(im_label, cv2.COLOR_GRAY2RGB)
    rgb_pred = cv2.cvtColor(im_pred, cv2.COLOR_GRAY2RGB)
    
    rgb_label[:,:,1:3] = 0*rgb_label[:,:,1:2]
    rgb_pred[:,:,0] = 0*rgb_pred[:,:,0]
    rgb_pred[:,:,2] = 0*rgb_pred[:,:,2]
    
    img_pred = cv2.addWeighted(rgb_img, 1, rgb_pred, 0.99, 0 )
    #img_label = cv2.addWeighted(rgb_img, 1, rgb_label, 0.3, 0)
    cv2.imwrite("./AOI/r/%d.jpg"%(i), img_pred)
    cv2.imwrite("./AOI/r/p%d.jpg"%(i), rgb_pred)

model = load_model('unet.h5')
X_test = util.load_test_images('./AOI/n/', img_type='jpg', img_format='gray', resize=(512, 512), ellipse=False)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=4)
predict = model.predict(X_test)

for i in range(predict.shape[0]):
    predict_evaluation(predict[i,:,:,0], X_test[i,:,:,0], X_test[i,:,:,0],i)


