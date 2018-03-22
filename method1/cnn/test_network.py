# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# python test_network.py --model santa_not_santa.model --folder testingData/c0

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os

IMG_WIDTH = 101 #128
IMG_HEIGHT = 179#157 #128


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
ap.add_argument("-f", "--folder", required=True,
	help="path to input image folder")

args = vars(ap.parse_args())


# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

def run():
	# 				 	 	T			F
	#	Positive		TP		FP
	#	Negative		FN		TN

	# precision: P = TP / (TP + FP)
	# recall:		 R = TP / (TP + FN)
	# F1 = 2PR / (P + R)

	TP=FP=FN=TN=0
	categories = os.listdir(args['folder'])
	for c in categories:
		files = os.listdir(args['folder']+'/'+c)
		for f in files:
			correct, error = classify(args['folder']+'/'+c+'/'+f)
			r = 'PASS' if correct > error else 'NG  '
			print('{0}/{1:35}, {2}, correct={3:3f}, error={4:3f}'.format(c, f, r, correct, error))
			if correct > error:
				if c == '0':
					TP += 1
				else:
					FP += 1
			else:
				if c == '0':
					FN += 1
				else:
					TN += 1

	# result	
	P = TP / (TP + FP)
	R = TP / (TP + FN)
	F1 = 2*P*R / (P + R)
	print("Precison={0}, Recall={1}, F1={2}".format(P, R, F1))



def classify(img):
	# load the image
	image = cv2.imread(img)
	orig = image.copy()

	# pre-process the image for classification
	image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# classify the input image
	# (notSanta, santa) = model.predict(image)[0]

	(correct, error) = model.predict(image)[0]
	return correct, error

	# (a, b, c) = model.predict(image)[0]

def output(correct, error):
	# build the label
	# label = "Santa" if santa > notSanta else "Not Santa"
	# proba = santa if santa > notSanta else notSanta
	# label = "{}: {:.2f}%".format(label, proba * 100)

	label = "correct" if correct > error else "error"
	proba = correct if correct > error else error

	# if a>b and a>c:
	# 	label, proba = 'a', a
	# elif b>a and b>c:
	#     label, proba = 'b', b
	# elif c>a and c>b:
	#     label, proba = 'c', c

	label = "{}: {:.2f}%".format(label, proba * 100)

	print(label)

	# # draw the label on the image
	# output = imutils.resize(orig, width=400)
	# cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	# 	0.7, (0, 255, 0), 2)

	# # show the output image
	# cv2.imshow("Output", output)
	# cv2.waitKey(0)

run()