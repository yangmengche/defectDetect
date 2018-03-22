import cv2 as cv

def byOrb(name):
  img = cv.imread(name)
  orb = cv.ORB_create()
  kp = orb.detect(img, None)
  kp, des = orb.compute(img, kp)
  img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
  cv.imshow(name, img2)
  cv.waitKey(0)

def byHarris(name):
  img = cv.imread(name)
  img8 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
  # cv.imshow('gray', img8)

  harrisStrength = cv.cornerHarris(img8, 7, 7, 0.2)
  thresh = 0.0001
  retval, cor = cv.threshold(harrisStrength, thresh, 255, cv.THRESH_BINARY)
  img2 = cv.drawKeypoints(img, cor, None, color=(0, 255, 0), flags=0)
  cv.imshow(name, cor)
  cv.waitKey(0)

def findCorner(filePath):
  # byOrb(filePath)
  # return

  byHarris(filePath)

  # kernal = np.ones((25, 25), np.uint8)
  # dia = cv.dilate(img, kernal)
  # cv.imshow('dilate', dia)
  # ero = cv.erode(dia, kernal)
  # cv.imshow('erosion', ero)
# end function