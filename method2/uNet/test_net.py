import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output folder")
ap.add_argument("-i", "--input", required=True, help="path to input folder of testing data")
ap.add_argument("-m", "--model", required=True, help="path to model")

args = vars(ap.parse_args())

import os
import shutil
if os.path.exists(args['output']):
  shutil.rmtree(args['output'])
os.mkdir(args['output'])
os.mkdir('{0}/{1}'.format(args['output'], 'TP'))
os.mkdir('{0}/{1}'.format(args['output'], 'FP'))
os.mkdir('{0}/{1}'.format(args['output'], 'FN'))
os.mkdir('{0}/{1}'.format(args['output'], 'TN'))

import tensorflow as tf
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
from keras import backend as K
import gc

import sys
# sys.path.append("./")
import utils
import iou

posPath = os.path.join(args['input'], 'pos')
negPath = os.path.join(args['input'], 'neg')
# X_test_p, y_test_p = utils.load_images_masks(
#   posPath,
#   img_type='jpg',
#   img_format='None',
#   resize=(512, 512),
#   ellipse=False)
# X_test_n, y_test_n = utils.load_images_masks(
#   negPath,
#   img_type='jpg',
#   img_format='None',
#   resize=(512, 512),
#   ellipse=False)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
def predict_evaluation(pred, image, label):
  '''
  '''
  # y_true = tf.constant(label, dtype=tf.float64)
  # y_pred = tf.constant(pred, dtype=tf.float64)
  # calIOU = utils.IOU_calc_loss(y_true, y_pred)
  # sess = tf.Session()
  # iou = -sess.run(calIOU)
  # transform gray image to rgb
  img = np.array(image, np.uint8)
  rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  del img
  # rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  # scale pred and mask's pixel range to 0~255
  # im_label = np.array(255 * label, dtype=np.uint8)
  im_pred = np.array(255 * pred, dtype=np.uint8)
  im_pred2 = cv2.morphologyEx(im_pred, cv2.MORPH_OPEN, kernel)
  mark1 = np.count_nonzero(im_pred)
  mark2 = np.count_nonzero(im_pred2)

  # transform both of them to rgb
  # rgb_label = cv2.cvtColor(im_label, cv2.COLOR_GRAY2RGB)
  rgb_pred = cv2.cvtColor(im_pred2, cv2.COLOR_GRAY2RGB)
  del im_pred, im_pred2

  # rgb_label[:, :, 1:3] = 0 * rgb_label[:, :, 1:2]
  rgb_pred[:, :, 0] = 0 * rgb_pred[:, :, 0]
  rgb_pred[:, :, 2] = 0 * rgb_pred[:, :, 2]

  img_pred = cv2.addWeighted(rgb_img, 1, rgb_pred, 0.99, 0)
  #img_label = cv2.addWeighted(rgb_img, 1, rgb_label, 0.3, 0)
  # cv2.imwrite("{0}/{1}.jpg".format(args['output'], i), img_pred)
  # cv2.imwrite("{0}/{1}_pred.jpg".format(args['output'], i), rgb_pred)
  # cv2.imwrite("{0}/{1}_src.jpg".format(args['output'], i), img)
  return mark1, 0, img_pred, rgb_pred, rgb_img, mark2

def outputImage(path, name, img_pred, rgb_pred, img):
  cv2.imwrite("{0}/{1}.jpg".format(path, name), img_pred)
  cv2.imwrite("{0}/{1}_pred.jpg".format(path, name), rgb_pred)
  cv2.imwrite("{0}/{1}_src.jpg".format(path, name), img)  

model = load_model(
  args['model'],
  custom_objects={
    'IOU_calc_loss': iou.IOU_calc_loss,
    'IOU_calc': iou.IOU_calc
  })
smooth = 1.

def batchPredict(testSet, labelSet, bPositive, offset):
  predict_result = model.predict(testSet)
  sum = tp = fp = fn = tn = 0
  print('evaluating...')
  for i in range(predict_result.shape[0]):
    name = '{0}{1}'.format('p' if bPositive else 'n', offset+i)
    mark1, iou, img_pred, rgb_pred, img, mark2 = predict_evaluation(predict_result[i, :, :, 0], testSet[i, :, :, :], labelSet[i, :, :, 0])
    mark = mark2
    if bPositive:
      if (mark > 100):
        tp += 1
        r = 'TP'
        # outputImage('{0}/{1}'.format(args['output'], r), name, img_pred, rgb_pred, img)
      else:
        fn += 1
        r = 'FN'
        outputImage('{0}/{1}'.format(args['output'], r), name, img_pred, rgb_pred, img)
    else:
      if (mark > 100):
        fp += 1
        r = 'FP'
        outputImage('{0}/{1}'.format(args['output'], r), name, img_pred, rgb_pred, img)
      else:
        tn += 1
        r = 'TN'
        # outputImage('{0}/{1}'.format(args['output'], r), name, img_pred, rgb_pred, img)
    print('{0}:{1}, mark1={2}, mark2={3}, iou={4:.4f}'.format(name, r, mark1, mark2, iou))
    sum += iou
    del img_pred, rgb_pred, img
  del predict_result
  return (sum, tp, fp, fn, tn)

BATCH = 25
width = height = 512
IMAGE_CHANNELS = 3
def test(bPositive, resize):
  sum = tp = fp = fn = tn = total = 0
  l = r = 0
  path_to_images = posPath if bPositive else negPath
  image_names = [x for x in os.listdir(path_to_images) if x.endswith('.jpg')]
  while True:
    r = r+BATCH if r+BATCH < len(image_names) else len(image_names)
    if l == r:
      break    
    images_all = np.empty([r-l, resize[0], resize[1], IMAGE_CHANNELS])
    labels_all = np.empty([r-l, resize[0], resize[1], 1])
    print('loading...')
    for image_index in range(l, r):
      image_filename = image_names[image_index]
      img, label = utils.load_one_image_mask(os.path.join(path_to_images, image_filename), 
        img_format='None', resize=resize, ellipse=False)
      images_all[image_index-l] = img
      labels_all[image_index-l] = label
    print('predicting...')
    ret = batchPredict(images_all, labels_all, bPositive, l)
    sum += ret[0]
    tp += ret[1]
    fp += ret[2]
    fn += ret[3]
    tn += ret[4]
    l = r
    del images_all
    del labels_all
    gc.collect()
  return (sum, tp, fp, fn, tn, len(image_names))
# predict_p = model.predict(X_test_p)
# predict_n = model.predict(X_test_n)

# sum = 0

# tp = fp = fn = tn = 0
# for i in range(predict_p.shape[0]):
#   name = 'p{0}'.format(i)
#   mark, iou, img_pred, rgb_pred, img = predict_evaluation(predict_p[i, :, :, 0], X_test_p[i, :, :, :], y_test_p[i, :, :, 0], name)
#   if (mark > 100):
#     tp += 1
#     r = 'TP'
#     outputImage('{0}/{1}'.format(args['output'], r), name, img_pred, rgb_pred, img)
#   else:
#     fn += 1
#     r = 'FN'
#     outputImage('{0}/{1}'.format(args['output'], r), name, img_pred, rgb_pred, img)
#   print('{0}:{1}, mark={2}, iou={3:.4f}'.format(name, r, mark, iou))
#   sum += iou
# if (len(predict_n) > 0):
#   for i in range(predict_n.shape[0]):
#     name = 'n{0}'.format(i)
#     mark, iou, img_pred, rgb_pred, img = predict_evaluation(predict_n[i, :, :, 0], X_test_n[i, :, :, :], y_test_n[i, :, :, 0], name)
#     if (mark > 100):
#       fp += 1
#       r = 'FP'
#       outputImage('{0}/{1}'.format(args['output'], r), name, img_pred, rgb_pred, img)
#     else:
#       tn += 1
#       r = 'TN'
#       outputImage('{0}/{1}'.format(args['output'], r), name, img_pred, rgb_pred, img)
    
#     print('{0}:{1}, mark={2}, iou={3:.4f}'.format(name, r, mark, iou))
#     sum += iou


result_p = test(True, (height, width))
result_n = test(False, (height, width))

sum = result_p[0] + result_n[0]
tp = result_p[1] + result_n[1]
fp = result_p[2] + result_n[2]
fn = result_p[3] + result_n[3]
tn = result_p[4] + result_n[4]
total = result_p[5] + result_n[5]
#	pred  \  real 	Pos		Neg
#	Positive		TP		FP
#	Negative		FN		TN

# precision: P = TP / (TP + FP)
# recall:	 R = TP / (TP + FN)
# F1 = 2PR / (P + R)
# total = len(predict_p) + len(predict_n)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = (2 * precision * recall) / (precision + recall)
print('       TP = {0:3}, FP= {1:3}'.format(tp, fp))
print('       FN = {0:3}, TN= {1:3}'.format(fn, tn))
print('precision = {0:.2f}%'.format(precision*100))
print('recall    = {0:.2f}%'.format(recall*100))
print('F1        = {0:.2f}%'.format(f1*100))
print('avg iou   = {0:.4F}'.format(sum / total))
