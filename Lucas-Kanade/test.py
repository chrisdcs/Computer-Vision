# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 21:13:09 2021

@author: m1390
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal as ss
from utils import normalize_img,Transform2D

img1 = cv2.imread(r'img\0001.jpg',0)
img2 = cv2.imread(r'img\0004.jpg',0)

img1 = normalize_img(img1)
img2 = normalize_img(img2)

x1,x2,y1,y2 = 70,150,160,210

T = img1[x1:x2,y1:y2]

# plt.imshow(T,cmap='gray')
# initialize W
W = np.column_stack([np.eye(2),np.zeros((2,1))])

X = np.array([x1,x2])
error = T - img2[x1:x2,y1:y2] 

Tr = Transform2D(img1)
IW = Tr.fit(x1, x2, y1, y2, W, img2)

d_x = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
d_y = d_x.T

# ∇I
Ix = ss.convolve2d(img2,d_x,'same')
Iy = ss.convolve2d(img2,d_y,'same')
# Ixy = Ix * Iy

# warp gradient of I (step 3)
IWx = Tr.fit(x1, x2, y1, y2, W, Ix)
IWy = Tr.fit(x1, x2, y1, y2, W, Iy)
IWxy = np.concatenate([IWx[:,:,None],IWy[:,:,None]],2)
gradI = np.zeros((2,IWx.shape[0]*IWx.shape[1]))
gradI[0] = IWx.flatten()
gradI[1] = IWy.flatten()
gradI = gradI.T

# compute Jacobian and Steepest Descent images
Jacobian = Tr.compute_jacobian(x1, x2, y1, y2, W)
SDI = (gradI[:,None,:] @ Jacobian).squeeze()

# compute Hessian and solve least squares
# H = SDI.T @ SDI
# delta_p = np.linalg.inv(H) @ (error.flatten() @ SDI)

Q,R = np.linalg.qr(SDI)
delta_p = np.linalg.inv(R) @ Q.T @ error.flatten()

# update parameters
W[0,0] += delta_p[0]
W[1,0] += delta_p[1]
W[0,1] += delta_p[2]
W[1,1] += delta_p[3]
W[0,2] += delta_p[4]
W[1,2] += delta_p[5]

IW_ = Tr.fit(x1, x2, y1, y2, W, img2)
plt.imshow(T,cmap='gray'),plt.show()
plt.imshow(img2[x1:x2,y1:y2],cmap='gray'),plt.show()
plt.imshow(IW_,cmap='gray'),plt.show()