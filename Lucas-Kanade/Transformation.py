# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:12:05 2021

@author: m1390
"""

import autograd as ad
# import autograd.numpy as np
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
#%% a short demo for linear transformation
img = cv2.imread('Lenna.png')

# extract image shape
h,w,c = img.shape

# define transformation matrix (in homogeneous coordinates)
theta = 0
t = .3
M = np.array([[np.cos(theta),-np.sin(theta),t],
              [np.sin(theta),np.cos(theta),t]]) # translation + rotation
# M = np.array([[1+1.4,0.5,0.],
#               [-0.5,1+1.4,0.]])

# generate normalized grid
x = np.linspace(-1,1,h)
y = np.linspace(-1,1,w)

xt,yt = np.meshgrid(x,y)
ones = np.ones(np.prod(xt.shape))
grid = np.vstack([xt.flatten(),yt.flatten(),ones])

grid_new = M @ grid
grid_new = grid_new.reshape(-1,h,w,order='F')

xt1 = grid_new[0]
yt1 = grid_new[1]

x = (xt1 + 1) * h / 2
y = (yt1 + 1) * w / 2

x0 = np.floor(x).astype(np.int64)
x1 = x0 + 1
y0 = np.floor(y).astype(np.int64)
y1 = y0 + 1

x0 = np.clip(x0,0,h-1)
y0 = np.clip(y0,0,w-1)
x1 = np.clip(x1,0,h-1)
y1 = np.clip(y1,0,w-1)

Ia = img[x0,y0]
Ib = img[x0,y1]
Ic = img[x1,y0]
Id = img[x1,y1]

# calculate deltas
wa = (x1-x) * (y1-y)
wb = (x1-x) * (y-y0)
wc = (x-x0) * (y1-y)
wd = (x-x0) * (y-y0)

# add dimension for addition
wa = np.expand_dims(wa, axis=2)
wb = np.expand_dims(wb, axis=2)
wc = np.expand_dims(wc, axis=2)
wd = np.expand_dims(wd, axis=2)

# compute output
out = wa*Ia + wb*Ib + wc*Ic + wd*Id

plt.imshow(out.astype(np.uint8))