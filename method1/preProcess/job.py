import cv2 as cv
import patternMatch as pm
import copy
import config
from pathlib import Path
import os

class Job:
  def __init__(self, folder, fileName, patterns):
    self.folder = folder
    self.file = fileName
    self.patterns = patterns
  def do(self):
    img24 = cv.imread('./image/{0}/{1}'.format(self.folder,self.file))
    if img24 is None:
      # raise IOError(' not found.')
      return
    img8 = cv.cvtColor(img24, cv.COLOR_BGR2GRAY)
    
    match = pm.findAllPattern(img8, self.patterns)
    
    match2 = []
    for category in match:
      seg = pm.segment(category[1], img24.shape[1])
      for s in seg:
        match2.append((category[0], s))

    # draw rect
    cr = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    buf2 = copy.copy(img24)

    # output pattern match result
    if config.mode == 'debug':
      for category in match:
        for pt in category[1]:
          cv.rectangle(img24, (pt[0], pt[1]), (pt[0]+pt[2], pt[1]+pt[3]), cr[category[0]], 1)
          cv.imwrite('./output/{0}/{1}_.jpg'.format(self.folder, self.file), img24)
        #end for
      #ned for
    #end if


    i=0
    for category in match2:
      dest = './output/{0}/{1}'.format(self.folder, category[0])
      if(not Path(dest).exists()):
        os.mkdir(dest)
      for pt in category[1]:
        # print('{0}/{1} rc={2}'.format(self.folder, self.file, pt))
        try:
          # output result
          if config.mode == 'debug':
            cv.rectangle(buf2, (pt[0], pt[1]), (pt[0]+pt[2], pt[1]+pt[3]), cr[category[0]], 1)
            cv.imwrite('./output/{0}/{1}'.format(self.folder, self.file), buf2)
          else:
            # output segment image files
            cv.imwrite('{0}/{1}_{2}-{3}-{4}.jpg'.format(dest, self.folder, self.file, category[0], i), img24[pt[1]:pt[1]+pt[3], pt[0]:pt[0]+pt[2]])
        except OverflowError:
          print('overflow: {0}/{1} rc={2}'.format(self.folder, self.file, pt))
          pass
        except IOError:
          print('IOerror: {0}/{1} rc={2}'.format(self.folder, self.file, pt))
        i+=1

    # cv.imshow('2', buf2)
    # cv.waitKey(0)    