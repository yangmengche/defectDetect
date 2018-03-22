import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='Create labeling dataset')
parser.add_argument('-s', '--source', type=str, required=True, 
  help='path to labeling data file')
args = vars(parser.parse_args())

workingFolder = os.path.abspath(args['source'])
dirs=['2', '20', '21', '22', '23', '24', '25']
output='{0}/all/'.format(workingFolder)
if not os.path.exists(output):
  os.mkdir(output)

for dir in dirs:
  src = os.path.join(workingFolder, dir)
  if os.path.exists(src):
    files = os.listdir(src)
    for file in files:
      if(os.path.isfile(src+'/'+file)):
        newName = '{0}-{1}'.format(dir, file)
        shutil.copy(src+'/'+file, output+newName)
