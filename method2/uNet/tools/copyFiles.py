import os
import shutil
import re

def copyFile(srcDir, outDir, name):
  src = os.path.join(srcDir, name)
  dest = os.path.join(outDir, name)
  if os.path.exists(src):
    shutil.copyfile(src, dest)
    return 1
  else:
    return 0


target = ['HKH740132H', 'HMH730006H', 'ICH760014H']

with open('./uNet/dust/notdust.txt', 'r') as reader:
  lines = reader.readlines()
  files = [x.strip() for x in lines]
  count=0
  total=0
  for d in target:
    src = os.path.join('.','testImage', d, 'all')
    for f in files:
      count += copyFile(src, os.path.join('.','uNet', d, 'test', 'dust','neg'), f)
  print('total={0}/{1}'.format(count, len(files)))