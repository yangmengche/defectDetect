import matplotlib.image as mpimg
import cv2 as cv
import numpy as np
import os

img = mpimg.imread(os.path.abspath('./uNet/ICH760014H/training/all/2-ICH760014H-010004.jpg'))
img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
mask = np.zeros_like(img)
cv.rectangle(mask, (0, 0), (10, 10), (255, 0, 0), thickness=cv.FILLED)

cv.imshow('a', mask)
shape = np.shape(mask)
label = np.reshape(mask, (shape[0], shape[1], 1))
cv.imshow('b', label)
cv.waitKey()

