
# coding: utf-8

# # Jupyter Notebook 101

# # Agenda
# 
# 1. Cell types (markdown/code)
# 2. Keyboard input modes (Edit/Command)
# 3. Some useful hotkeys
# 4. Check installed Library
# 5. Install library with pip
# 6. restart kernel / run all

# ## 1. Cell types

# - code cell
# - markdown cell

# In[1]:


print(1+2)


# In[2]:


for i in range(10):
    print(i)


# # headline1
# ## headline2
# ### headline3
# __bold__

# _italian_

# ## 2. keyboard input modes

# The Jupyter Notebook has two different keyboard input modes. Edit mode allows you to type code or text into a cell and is indicated by a green cell border. Command mode binds the keyboard to notebook level commands and is indicated by a grey cell border with a blue left margin.
# 
# - esc => blue (Command mode)
# - enter => green (Edit mode)

# ## 3. Some useful hotkeys

# ### hotkeys
# 
# - a - insert cell above
# - b - insert cell below
# - x - cut selected cell
# - z - undo cell deletion
# - h - show keyboard hotkeys (shortcut)
# - y - change cell to code cell
# - m - change cell to markdow cell
# - Shift-Enter - run cell, select below 

# ## 4. Check environment

# In[3]:


get_ipython().system(u'nvidia-smi ')


# In[4]:


get_ipython().system(u'pwd')


# In[4]:


get_ipython().system(u'ls -alh')


# ## 4. Check library versions

# In[6]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import sklearn


# In[7]:


print('tensorflow version: {}'.format(tf.__version__))
print('numpy version: {}'.format(np.__version__))
print('pandas version: {}'.format(pd.__version__))
print('matplotlib version: {}'.format(matplotlib.__version__))
print('scikit learning version: {}'.format(sklearn.__version__))


# ## 5. Import necessary libraries

# In[5]:


get_ipython().system(u'pip install keras')


# In[6]:


import keras


# In[7]:


print('keras version: {}'.format(keras.__version__))


# In[8]:


get_ipython().system(u'pip install opencv-python')


# In[9]:


get_ipython().system(u'apt update && apt install -y libsm6 libxext6')


# In[10]:


import cv2


# In[15]:


print('opencv version: {}'.format(cv2.__version__))


# In[12]:


get_ipython().system(u'pip install xmltodict')


# In[13]:


get_ipython().system(u'pip install -U scikit-learn')


# In[14]:


get_ipython().system(u'pip install h5py')

