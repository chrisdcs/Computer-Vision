# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:17:25 2021

Histogram equalization

@author: m1390
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
#%%
def histogram(img):
    # assume image from 0-255
    x = np.arange(0,256)
    hist = np.zeros(x.shape)
    cdf = np.zeros(x.shape)
    for i in x:
        hist[i] = np.sum(img == i)
        cdf[i] = np.sum(hist[:i+1])
    return hist, cdf
#%% read intensity image
img = cv2.imread('clahe.jpg',0)
plt.imshow(img), plt.show()
flat = img.flatten()
n = img.shape[0]*img.shape[1]
hist,cdf = histogram(img)
cdf = cdf/(cdf.max() - cdf.min())*255
img_new = cdf[flat].reshape(img.shape)
plt.imshow(img_new)