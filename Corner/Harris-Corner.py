# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 20:35:58 2021

@author: m1390
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.stats as st
#%%
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

# Sobel operator for computing image gradient
d_x = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
d_y = d_x.T

F_name = 'Lenna.png'#'box.jpg'
I = cv2.imread(F_name,0)
Ix = ss.convolve2d(I,d_x,'same')
Iy = ss.convolve2d(I,d_y,'same')
Ixy = Ix * Iy

Ixx = ss.convolve2d(Ix**2,gkern(5,1),'same')
Iyy = ss.convolve2d(Iy**2,gkern(5,1),'same')
Ixy = ss.convolve2d(Ixy,gkern(5,1),'same')

det = Ixx * Iyy - Ixy ** 2
trace = Ixx + Iyy

harris = det - 0.06*trace**2
harris = (harris - harris.mean())/harris.std()

corners = (harris > .3)
plt.imshow(corners)