import tkinter
from tkinter import ttk
from PIL import Image, ImageDraw
from PIL.ImageTk import PhotoImage
from tkinter.filedialog import askopenfilename
import os
import numpy as np
import matplotlib.image as mpimg
import json

from importlib.machinery import SourceFileLoader
utils = SourceFileLoader('utils', './utils.py')

class App(ttk.Frame):
  def __init__(self, parent, *args, **kwargs):
    ttk.Frame.__init__(self, parent, *args, **kwargs)
    self.root = parent
    self.init_gui()
    self.lbl = tkinter.Label()
    self.index = 0
    parent.bind('<Left>', self.leftKey)
    parent.bind('<Right>', self.rightKey)

  def openFile(self, name):
    img = mpimg.imread(name, format='jpg')
    path_to_json = name.replace('jpg', 'json')
    if os.path.exists(path_to_json):
      with open(path_to_json) as json_file:
        json_get = json.load(json_file)
        pts_list = [np.array(pts['points'], np.int32) for pts in json_get['shapes']]
    else:
      pts_list=[]
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for pts in pts_list:
      pts = [tuple(pt) for pt in pts]
      draw.polygon(pts, outline=(255, 0, 0))
    self.lbl.pimg = PhotoImage(img)
    self.lbl.config(image=self.lbl.pimg)
    self.lbl.pack()
    size = '{0}x{1}'.format(img.size[0], img.size[1])
    self.root.geometry(size)
    self.root.title('{0}/{1} - {2}'.format(self.index+1, len(self.list), name))

  def on_Open(self):
    name = askopenfilename(initialdir="./",
                           filetypes =(("Image File", "*.jpg"), ("All", "*.*")),
                           title = "Choose a file.")
    self.dir = os.path.dirname(name)
    self.list = [x for x in os.listdir(self.dir) if x.endswith('.jpg')]
    self.list.sort()
    self.openFile(name)
    self.index = self.list.index(os.path.basename(name))
    
  def init_gui(self):
    self.menubar = tkinter.Menu(self.root)
    self.menubar.add_command(label='Open', command=lambda: self.on_Open())
    self.root.config(menu=self.menubar)


mainFrame = tkinter.Tk()
mainFrame.geometry('1024x768+400+100')
App(mainFrame)
mainFrame.mainloop()
