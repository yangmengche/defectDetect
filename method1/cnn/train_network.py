# USAGE
# python train_network.py --dataset images --model santa_not_santa.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from pyimagesearch.lenet import LeNet
from pyimagesearch.unet import UNet
from pyimagesearch.resnet import ResnetBuilder
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from pathlib import Path

# limit the max thread
import tensorflow as tf
from keras import backend as k
NUM_THREADS=8
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))
k.set_session(sess)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to output model")
# ap.add_argument("-p", "--plot", type=str, default="plot.png",
# 	help="path to output loss/accuracy plot")
ap.add_argument("-e", "--epochs", type=int, default=25,
	help="How many epochs, default=25")	
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = args['epochs']
INIT_LR = 1e-3
BS = 32
IMG_WIDTH = 101 #128
IMG_HEIGHT = 179 #157 #128
NUM_CLASSES = 2
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)



def training(dataSet, modelName):
	csv_logger = CSVLogger('{0}.csv'.format(modelName))
	# initialize the data and labels
	print("[INFO] loading images...")
	data = []
	labels = []

	# grab the image paths and randomly shuffle them
	# imagePaths = sorted(list(paths.list_images(args["dataset"])))
	imagePaths = sorted(list(paths.list_images(dataSet)))

	random.seed(42)
	random.shuffle(imagePaths)

	# loop over the input images
	for imagePath in imagePaths:
		# load the image, pre-process it, and store it in the data list
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
		image = img_to_array(image)
		data.append(image)

		# extract the class label from the image path and update the
		# labels list
		label = imagePath.split(os.path.sep)[-2]

		# Biclass
		# label = 1 if label == "santa" else 0
		label = 1 if label == "2" else 0
		
		# Multiple classes
		# category = 0
		# if label == 'a':
			# 		category = 0
		# if label == 'b':
			# 		category = 1
		# if label == 'c':
			# 		category = 2


		tmp = []
		tmp.append(label)
		# tmp.append(category)
		labels.append(tmp)

	# scale the raw pixel intensities to the range [0, 1]
	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)

	# partition the data into training and testing splits using 75% of
	# the data for training and the remaining 25% for testing
	(trainX, testX, trainY, testY) = train_test_split(data,
		labels, test_size=0.25, random_state=42)

	# convert the labels from integers to vectors
	trainY = to_categorical(trainY, num_classes=NUM_CLASSES)
	testY = to_categorical(testY, num_classes=NUM_CLASSES)

	# construct the image generator for data augmentation
	aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
		height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
		horizontal_flip=True, fill_mode="nearest")

	# initialize the model
	print("[INFO] compiling model...")
	# LeNet
	# model = LeNet.build(width=IMG_WIDTH, height=IMG_HEIGHT, depth=3, classes=NUM_CLASSES)
	# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	# model.compile(loss="binary_crossentropy", optimizer=opt,
	# 	metrics=["accuracy"])

	# U-Net
	# model = UNet()
	# model = model.build(width=IMG_WIDTH, height=IMG_HEIGHT, depth=3, classes=NUM_CLASSES)
	# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	# model.compile(loss="binary_crossentropy", optimizer=opt,
	# 	metrics=["accuracy"])

	# ResNet
	model = ResnetBuilder.build_resnet_152((3, IMG_WIDTH, IMG_HEIGHT), NUM_CLASSES)
	model.summary()
	model.compile(loss='categorical_crossentropy',
								optimizer='adam',
								metrics=['accuracy'])

	# train the network
	print("[INFO] training network...")
	print('[INFO] Dateset={0}, model name={1}'.format(dataSet, modelName))
	# LeNet
	# H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	# 	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	# 	epochs=EPOCHS, verbose=1)

	# U-Net
	# H = model.fit(trainX,trainY,validation_data=(testX,testY),epochs=10,batch_size=200,verbose=2)

	# ResNet
	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=True,  # randomly flip images
		vertical_flip=False)  # randomly flip images

	# Compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(trainX)

	# Fit the model on the batches generated by datagen.flow().
	H = model.fit_generator(datagen.flow(trainX, trainY, batch_size=BS),
						steps_per_epoch=trainX.shape[0] // BS,
						validation_data=(testX, testY),
						epochs=EPOCHS, verbose=1, max_q_size=100,
						callbacks=[lr_reducer, early_stopper, csv_logger])

	# save the model to disk
	print("[INFO] serializing network...")
	# model.save(args["model"])
	model.save(modelName)
	

	# plot the training loss and accuracy
	# plt.style.use("ggplot")
	# plt.figure()
	# N = EPOCHS
	# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	# plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	# plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	# plt.title("Training Loss and Accuracy on Santa/Not Santa")
	# plt.xlabel("Epoch #")
	# plt.ylabel("Loss/Accuracy")
	# plt.legend(loc="lower left")
	# plt.savefig(args["plot"])

categorise = os.listdir(args["dataset"])
for c in categorise:
	p = "{0}/{1}".format(args["dataset"], c)
	if Path(p).is_dir:
		training(p, c+'.model')
