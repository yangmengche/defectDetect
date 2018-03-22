import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--predict", required=True,
  help="path to predict csv file")
ap.add_argument("-e", "--expect", required=True,
  help="path to expect csv file")  
args = vars(ap.parse_args())

import csv
import os

# Expect FORMAT: tag, filename, sourceFolder
expect={}
with open(os.path.abspath(args['expect']), 'r') as expectFile:
  expectData = csv.reader(expectFile, delimiter=',')
  for e in expectData:
    expect[e[1].strip()]=e[0].strip()

#       dust   leak  normal
#dust   TP      FP    FP
#leak   FN      TP    FP
#normal FN      FN    TN

def evaluation(expect, predict, tag):
  if expect == tag:
    if predict == tag:
      return 'tp'
    else:
      return 'fn'
  else:
    if predict == tag:
      return 'fp'
    else:
      return 'tn'

result = {
  'tp': 0,
  'fp': 0,
  'fn': 0,
  'tn': 0
}
precision = recall = f1 = 0
# Predict FORMAT: filename, tag, mark
with open(os.path.abspath(args['predict']), 'r') as predictFile:
  predictData = list(csv.reader(predictFile, delimiter=','))
  predictData.sort(key = lambda item: item[0])
  for p in predictData:
    if p[0] in expect:
      r = evaluation(expect[p[0]], p[1], 'dust')
      if r in ['tn', 'fn']:
        r = evaluation(expect[p[0]], p[1], 'dust')
      result[r] += 1
      if r in ['fp', 'fn']:
        print('{0}, {1}'.format(r, p))

precision = result['tp'] / (result['tp'] + result['fp'])
recall = result['tp'] / (result['tp'] + result['fn'])
f1 = 2 * precision * recall / (precision + recall)

print('       TP = {0:3}, FP= {1:3}'.format(result['tp'], result['fp']))
print('       FN = {0:3}, TN= {1:3}'.format(result['fn'], result['tn']))
print('precision = {0:.2f}%'.format(precision*100))
print('recall    = {0:.2f}%'.format(recall*100))
print('F1        = {0:.2f}%'.format(f1*100))