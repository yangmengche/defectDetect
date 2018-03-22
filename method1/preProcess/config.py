# parameter
maxThread = 8
patterns = [
  {
    'name': 'pattern1.png',
    # 'threshold': 0.2
    'threshold': 0.75
  },
  {
    'name': 'pattern2.png',
    # 'threshold': 0.02 #SQDIFF
    'threshold': 0.94 #CCORR
  }
  # {
  #   'name': 'pattern3.png',
  #   # 'threshold': 0.2
  #   'threshold': 0.50
  # }  
]

# ['normal', 'debug']
mode = 'normal'