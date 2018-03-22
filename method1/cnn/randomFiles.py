import random
import os
import shutil
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--src", required=True,
	help="path to input dataset")
ap.add_argument("-d", "--dest", required=True,
	help="path to output dataset")
ap.add_argument("-q", "--quantity", required=True, type=int,
	help="How many items to move to destination")
ap.add_argument("-m", "--move", type=bool, default=False,
	help="move or copy?")  
args = vars(ap.parse_args())

source = args['src']
dest = args['dest']
q = args['quantity']
isMove = args['move']

if os.path.exists(dest):
  shutil.rmtree(dest)

def choose(input ,output):
  os.makedirs(output)
  items = os.listdir(input)
  # handle folder
  srcFiles = []
  for i in items:
    if(Path(input+'/'+i).is_dir()):
      choose(input+'/'+i, output+'/'+i)
    else:
      srcFiles.append(i)
  # random choose files
  destFiles= []
  if(len(srcFiles) < q):
    print('files in {0} is less than {1}'.format(input, q)) 
    return
  while len(destFiles) < q:
    f = random.choice(srcFiles)
    if f in destFiles:
      continue
    else:
      destFiles.append(f)
      if(isMove):
        shutil.move(input+'/'+f, output+'/'+f)    
      else:
        shutil.copy(input+'/'+f, output+'/'+f)

choose(source, dest)