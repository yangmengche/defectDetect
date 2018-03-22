import cv2 as cv
import copy

def findPattern(img, pattern, threshold, pw, ph):
  match = []
  count=0
  while 1 or count > 1000:
    # val, t = finePattern(img8, pattern8)
    result = cv.matchTemplate(img, pattern, cv.TM_CCOEFF_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
    val = maxVal
    loc = maxLoc
    if(val < threshold):
      break
    # t = (minLoc[0], minLoc[1])
    t = (loc[0], loc[1], pw, ph)
    match.append(t)
    # print(t)
    # mask src image
    cv.rectangle(img, (t[0], t[1]), (t[0]+pw, t[1]+ph), (0, 0, 0), -1)
    count +=1
  #endwhile
  return match
# end funciton

def findAllPattern(img, patterns):
  try:
    match = []
    id=0
    for p in patterns:
      clone = copy.copy(img)
      pw = p['img'].shape[1]
      ph = p['img'].shape[0]
      # print('*******{0}*********'.format(id))
      ret = findPattern(clone, p['img'], p['threshold'], pw, ph)
      match.append(('[p{0}]'.format(id), ret))
      id+=1
      # flip image
      p2 = cv.flip(p['img'], 1)
      # print('*******{0}*********'.format(id))
      ret = findPattern(clone, p2, p['threshold'], pw, ph)
      match.append(('[p{0}]'.format(id), ret))
      id +=1
    return match
  except Exception as e:
    print(e)


def segment(match, boundary):
  if len(match) ==0:
    return []
  # group, keep (y, range:height/2, [pt], gap)
  init = 99999999999
  # group=[{'y': match[0][1], 'range':match[0][3]/2, 'pt':[], 'gap':init, 'w':match[0][2], 'h': match[0][3]}]
  group=[]
  for pt in match:
    done=False
    for g in group:
      if abs(pt[1]-g['y']) < g['range']:
        g['pt'].append(pt)
        done=True
        break
    if not done:
      # not in any group, create new
      group.append({'y':pt[1], 'range':pt[3]/2, 'pt':[pt], 'gap':init, 'w':pt[2], 'h': pt[3]})

  # find gap for all group
  for g in group:
    if len(g['pt']) > 1:
      g['pt'].sort(key = lambda ele: ele[0])
      compare = g['pt'][0][0]
      for pt in g['pt'][1:]:
        diff = abs(pt[0]-compare)
        if g['gap'] > diff:
          g['gap'] = diff
        compare = pt[0]
      #endfor
    else:
      group.remove(g)
  #endfor
  match2=[]
  for g in group:
    if g['gap'] > g['w']*1.5:
      width = g['w']
    else:
      width = g['gap']
    boundary -= width
    base = g['pt'][len(g['pt']) // 2]
    x = base[0] - width * (base[0] // width)
    group2=[]
    while 1:
      group2.append((x, base[1], width, base[3]))
      x += width
      if x > boundary:
        match2.append(group2)
        break
    # end while
  # endfor
  return match2
# end function