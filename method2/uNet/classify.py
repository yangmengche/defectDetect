import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="<Required> Path to output folder")
ap.add_argument("-i", "--input", required=True, help="<Required> Path to input folder of testing data")
ap.add_argument('-m','--models', nargs='+', help='<Required> Model list, The size of model and tag must be the same', required=True)
ap.add_argument('-t','--tags', nargs='+', help='<Required> Tag list', required=True)

args = vars(ap.parse_args())

import os
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
import csv
csvPath = os.path.abspath('{0}/predict.csv'.format(args['output']))
print('csv={0}'.format(csvPath))
if os.path.exists(csvPath):
  os.remove(csvPath)
# with open(csvPath, 'w', newline='') as csvFile:
#   writer = csv.writer(csvFile)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
def predict_evaluation(pred, image):
  '''
  '''
  # y_pred = tf.constant(pred, dtype=tf.float64)
  # transform gray image to rgb
  img = np.array(image, np.uint8)
  rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  del img
  # rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  # scale pred and mask's pixel range to 0~255
  im_pred = np.array(255 * pred, dtype=np.uint8)
  im_pred2 = cv2.morphologyEx(im_pred, cv2.MORPH_OPEN, kernel)
  # mark = np.count_nonzero(im_pred)
  mark2 = np.count_nonzero(im_pred2)

  # transform both of them to rgb
  rgb_pred = cv2.cvtColor(im_pred2, cv2.COLOR_GRAY2RGB)
  del im_pred, im_pred2

  rgb_pred[:, :, 0] = 0 * rgb_pred[:, :, 0]
  rgb_pred[:, :, 2] = 0 * rgb_pred[:, :, 2]

  img_pred = cv2.addWeighted(rgb_img, 1, rgb_pred, 0.99, 0)

  return mark2, img_pred, rgb_pred, rgb_img

def outputImage(path, name, img_pred, rgb_pred, img):
  cv2.imwrite("{0}/{1}.jpg".format(path, name), img_pred)
  cv2.imwrite("{0}/{1}_pred.jpg".format(path, name), rgb_pred)
  cv2.imwrite("{0}/{1}_src.jpg".format(path, name), img)  

def removeImage(path, name):
  if os.path.exists("{0}/{1}.jpg".format(path, name)):
    os.remove("{0}/{1}.jpg".format(path, name))
    os.remove("{0}/{1}_pred.jpg".format(path, name))
    os.remove("{0}/{1}_src.jpg".format(path, name))

def batchPredict(model, names, testSet, output, offset, ret):
  predict_result = model.predict(testSet)
  # plist=[]
  # nlist=[]
  # p = n = 0
  result={}
  print('evaluating...')
  for i in range(predict_result.shape[0]):
    name = names[i]
    mark, img_pred, rgb_pred, img = predict_evaluation(predict_result[i, :, :, 0], testSet[i, :, :, :])

    if not ret:
      if mark > 100:
        r = output['p']
        outputImage(os.path.join(output['root'], output['p']), os.path.splitext(name), img_pred, rgb_pred, img)
      else:
        r = output['n']
        outputImage(os.path.join(output['root'], output['n']), os.path.basename(name), img_pred, rgb_pred, img)
    else:
      if mark > 100 and mark > ret[name][0]:
        r = output['p']
        if ret:
          removeImage(os.path.join(output['root'], ret[name][1]), os.path.splitext(name))
        outputImage(os.path.join(output['root'], output['p']), os.path.splitext(name), img_pred, rgb_pred, img)
      else:
        r = ret[name][1]
        # outputImage(os.path.join(output['root'], output['n']), os.path.basename(name), img_pred, rgb_pred, img)


    # if mark > 100:
    #   if not ret or mark > ret[name][0]:
    #     # p += 1
    #     # plist.append(name)
    #     if 'p' in output:
    #       r = output['p']
    #       if ret:
    #         removeImage(os.path.join(output['root'], ret[name][1]), os.path.basename(name))
    #       outputImage(os.path.join(output['root'], output['p']), os.path.basename(name), img_pred, rgb_pred, img)
    #       # writer.writerow([name, r, mark])
    #     else:
    #       r = 'p'
    #   else:
    #     del img_pred, rgb_pred, img
    #     continue
    # else:
    #   # n +=1
    #   # nlist.append(name)
    #   if 'n' in output:
    #     r = output['n']
    #     outputImage(os.path.join(output['root'], output['n']), os.path.basename(name), img_pred, rgb_pred, img)
    #     # writer.writerow([name, r, mark])
    #   else:
    #     r = 'n'
    result[name]=(mark, r)
    print('{0}, classify={1}, mark={2}'.format(name, r, mark))
    del img_pred, rgb_pred, img
  del predict_result
  # return (p, n, plist, nlist)
  return result

BATCH = 25
width = height = 512
IMAGE_CHANNELS = 3
# input: file list of test set
# output: path to output set : {p: path to positive set, n: path to negative set}
def test(modelPath, intputPath, inputs, output, resize, ret):
  model = load_model(
    modelPath,
    custom_objects={
      'IOU_calc_loss': iou.IOU_calc_loss,
      'IOU_calc': iou.IOU_calc
    })
  # plist = []
  # nlist = []
  # p = n = 0
  result={}
  l = r = 0
  while True:
    r = r+BATCH if r+BATCH < len(inputs) else len(inputs)
    if l == r:
      break    
    images_all = np.empty([r-l, resize[0], resize[1], IMAGE_CHANNELS])
    names = []
    print('loading...')
    for image_index in range(l, r):
      image_filename = inputs[image_index]
      img, label = utils.load_one_image_mask(os.path.join(intputPath, image_filename), 
        img_format='None', resize=resize, ellipse=False)
      images_all[image_index-l] = img
      names.append(image_filename)

    print('predicting...')
    result.update(batchPredict(model, names, images_all, output, l, ret))
    # p += ret[0]
    # n += ret[1]
    # plist += ret[2]
    # nlist += ret[3]
    l = r
    del images_all
    gc.collect()
  # return (p, n, plist, nlist, len(inputs))
  return result

#setup
import shutil
if os.path.exists(args['output']):
  shutil.rmtree(args['output'])
os.mkdir(args['output'])
for t in args['tags']:
  os.mkdir('{0}/{1}'.format(args['output'], t))
os.mkdir('{0}/{1}'.format(args['output'], 'normal'))

#run
def run(task):
  modelPath = task['modelPath']
  image_names = task['names']
  output = {
    'root': task['root'],
    'p': task['p'],
    'n': task['n']
  }  
  # if 'n' in task: 
  #   output['n'] = task['n']
  return test(modelPath, args['input'], image_names, output, (height, width), task['ret'] if 'ret' in task else None)

def runTasks(tasks):
  ret = None
  for t in tasks:
    if ret:
      t['ret'] = ret
    ret = run(t)
  return ret

tasks = []
for i in range(len(args['models'])):
  tasks.append({
    'names': [x for x in os.listdir(args['input']) if x.endswith('.jpg')],
    'root': args['output'],
    'modelPath':args['models'][i],
    'p': args['tags'][i],
    'n': 'normal'
  })
# # first
# tasks.append({
#   'names': [x for x in os.listdir(args['input']) if x.endswith('.jpg')],
#   'root': args['output'],
#   'modelPath':args['models'][0],
#   'p': args['tags'][0],
# })
# # mid
# last = len(args['models'])-1
# for i in range(1,last):
#   tasks.append({
#     'root': args['output'],
#     'modelPath':args['models'][i],
#     'p': args['tags'][i],
#   })
# #last
# tasks.append({
#   'root': args['output'],
#   'modelPath':args['models'][last],
#   'p': args['tags'][last],
#   'n':'normal'
# })
ret = runTasks(tasks)

with open(csvPath, 'w', newline='') as csvFile:
  writer = csv.writer(csvFile)
  for name, predict in ret.items():
    writer.writerow([name, predict[1], predict[0]])
print('done!!')