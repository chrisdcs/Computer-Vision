# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 21:13:09 2021

@author: m1390
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import normalize_img

img1 = cv2.imread(r'img\0001.jpg',0)
img2 = cv2.imread(r'img\0005.jpg',0)

img1 = normalize_img(img1)
img2 = normalize_img(img1)

x1,x2,y1,y2 = 70,150,150,220

T = img1[x1:x2,y1:y2]

plt.imshow(T,cmap='gray')
# initialize W
W = np.column_stack([np.eye(2),np.zeros((2,1))])

X = np.array([x1,x2])
error = T - img2[x1:x2,y1:y2] 
