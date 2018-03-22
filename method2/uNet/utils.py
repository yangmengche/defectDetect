import matplotlib.image as mpimg
import cv2 as cv
import os
import json 
import numpy as np


IMAGE_CHANNELS = 3

# return X, y
def get_mask_seg_polygon(path_to_img, gray=False):
  img = mpimg.imread(path_to_img)
  
  if not gray:
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
  
  basename = os.path.basename(path_to_img)
  
  path_to_json = path_to_img.replace('jpg', 'json')
  
  # print(path_to_json)
  
  if os.path.exists(path_to_json):
    with open(path_to_json) as json_file:
      json_get = json.load(json_file)
      pts_list = [np.array(pts['points'], np.int32) for pts in json_get['shapes']]
  else:
    pts_list=[]
  
  
  # return np.array(pts, np.int32)
  
  mask = np.zeros_like(img)
  
  # rgb_pred = cv2.cvtColor(im_pred, cv2.COLOR_GRAY2RGB)
  # return mask
  mask = cv.fillPoly(mask, pts_list, (255, 255, 255))
  mask[mask > 0] = 1.
  
  return mask

def get_mask_seg_polygon_multi(path_to_img, gray=False):
  img = mpimg.imread(path_to_img)
  
  if not gray:
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
  
  basename = os.path.basename(path_to_img)
  
  path_to_json = path_to_img.replace('jpg', 'json')
  
  # print(path_to_json)
  
  labels={}
  if os.path.exists(path_to_json):
    with open(path_to_json) as json_file:
      json_get = json.load(json_file)
      # pts_list = [np.array(pts['points'], np.int32) for pts in json_get['shapes']]
      for pts in json_get['shapes']:
        pts_list = [np.array(pts['points'], np.int32) ]
        if pts['label'] in labels:
          labels[pts['label']] += (pts_list)
        else:
          labels[pts['label']] = pts_list
  else:
    pts_list=[]
  
  
  # return np.array(pts, np.int32)
  
  mask = np.zeros_like(img)
  
  # rgb_pred = cv2.cvtColor(im_pred, cv2.COLOR_GRAY2RGB)
  # return mask
  # mask = cv.fillPoly(mask, pts_list, (255, 255, 255))
  # mask[mask > 0] = 1.
  
  for k, pts_list in labels.items():
    color = int(k)
    cv.fillPoly(mask, pts_list, (color, color, color))
  
  return mask

def load_one_image_mask(img_path, img_format, resize, ellipse=False):
  image = mpimg.imread(img_path, format=img_format)
  if ellipse:
    mask = get_mask_seg_ellipse(img_path)
  else:
    mask = get_mask_seg_polygon(img_path)

  if resize:
    image = cv.resize(image, (resize[0], resize[1]))
    mask = cv.resize(mask, (resize[0], resize[1]))
  # image = np.reshape(image, (resize[0], resize[1], IMAGE_CHANNELS))
    label = np.reshape(mask, (resize[0], resize[1], 1))
  else:
    shape = np.shape(mask)
    label = np.reshape(mask, (shape[0], shape[1], 1))
  del mask
  return image, label

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
    img, label = load_one_image_mask(os.path.join(path_to_images, image_filename), img_format=img_format, resize=resize, ellipse=ellipse)
    images_all[image_index] = img
    labels_all[image_index] = label

#     image = mpimg.imread(os.path.join(path_to_images, image_filename), format=img_format)
#     if ellipse:
#       mask = get_mask_seg_ellipse(os.path.join(path_to_images, image_filename))
#     else:
#       mask = get_mask_seg_polygon(os.path.join(path_to_images, image_filename))

#     if resize:
#       image = cv.resize(image, (resize[0], resize[1]))
#       mask = cv.resize(mask, (resize[0], resize[1]))
# #print (image.shape,image_filename)
#     images_all[image_index] = np.reshape(image, (resize[0], resize[1], IMAGE_CHANNELS))
#     labels_all[image_index] = np.reshape(mask, (resize[0], resize[1], 1))

  return images_all, labels_all

