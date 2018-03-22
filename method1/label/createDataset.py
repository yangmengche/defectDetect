import argparse
import glob
import subprocess
import os

parser = argparse.ArgumentParser(description='Create labeling dataset')
parser.add_argument('-s', '--source', type=str, required=True, 
  help='path to labeling data file')
args = vars(parser.parse_args())

cmd = '/usr/local/bin/labelme_json_to_dataset {0} -o {1}/{2}'
files = glob.glob(args['source']+'/*.json')
for f in files:
  name = os.path.basename(f).split('.')[0]
  t = cmd.format(f, args['source'], name)
  print(t)
  subprocess.check_output(t, shell=True)


