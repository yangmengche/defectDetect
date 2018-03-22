import numpy as np
import os
import sys
import queue
import cornerDetect as cd
import job
import cv2 as cv
import threading
import time
import shutil
import config
# cd.findCorner('./image/HKH740132H-010003.jpg')
# findCorner('./res/HKH740132H-010056.jpg')
# findAllPattern('./res/HKH740132H-010056.jpg')

# sys.exit(0)



# init
if os.path.exists('./output'):
  shutil.rmtree('./output')
os.mkdir('./output')

for p in config.patterns:
  img24 = cv.imread('./pattern/'+p['name'])
  p['img'] = cv.cvtColor(img24, cv.COLOR_BGR2GRAY)
  p['h'] = img24.shape[0]
  p['w'] = img24.shape[1]

# pFiles = os.listdir('./pattern')
# for p in pFiles:
#   pimg = cv.imread('./pattern/'+p)
#   if pimg is not None:
#     pimg8 = cv.cvtColor(pimg, cv.COLOR_BGR2GRAY)
#     patterns.append({'img': pimg8, 'threshold': 0.2})

que = queue.Queue()
folders = os.listdir('./image')
for folder in folders:
  files = os.listdir('./image/'+folder)
  os.mkdir('./output/'+folder)
  for f in files:
    que.put(job.Job(folder, f, config.patterns))
# endfor

total = que.qsize()
def dispatch():
  while que.qsize() > 0:
    task = que.get()
    sys.stdout.write('{0}/{1} : {2}/{3} \r'.format(que.qsize(), total, task.folder, task.file))
    sys.stdout.flush()    
    task.do()

threads=[]
for i in range(0, config.maxThread):
  t = (threading.Thread(target=dispatch, name=i))
  t.start()
  threads.append(t)

wait = True
while wait:
  wait = False
  time.sleep(1)
  for t in threads:
    wait |= t.is_alive()

print('done!')

cv.destroyAllWindows()