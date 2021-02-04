# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:46:28 2021

@author: m1390
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
#%% normalization image function
def norm(img):
    img = (img - np.min(img))*255/(np.max(img)-np.min(img))
    return img.astype(np.uint8)

#%% gamma correction
img = cv2.imread('bird.jpg')
gamma = 3
img_g = norm(img**(1/gamma))
plt.imshow(img),plt.show()
plt.imshow(img_g), plt.show()
#%% color transform
parameters = np.random.rand(3)
img_c = img.copy()
for i in range(3):
    img_c[:,:,i] = img_c[:,:,i] * parameters[i]
    
# img_c = norm(img_c)
plt.imshow(norm(img_c**(1/gamma)))