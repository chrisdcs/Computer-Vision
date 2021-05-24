# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:41:28 2021

@author: m1390
"""

import cv2
import matplotlib.pyplot as plt
from utils.LucasKanade import Lucas_Kanade
from utils.miscellaneous import normalize_img



img1 = cv2.imread(r'img\0001.jpg',0)
img2 = cv2.imread(r'img\0004.jpg',0)

ratio = 0.3
# downsample the image to speed up computation
img1 = cv2.resize(img1,(int(img1.shape[1]*ratio),int(img1.shape[0]*ratio)))
img2 = cv2.resize(img2,(int(img2.shape[1]*ratio),int(img2.shape[0]*ratio)))

img1 = normalize_img(img1)
img2 = normalize_img(img2)

x1,x2,y1,y2 = 22,44,47,64

T = img1
max_iter = 50
tol = 1e-1

LKT = Lucas_Kanade(img1, max_iter, tol)
box, IW = LKT.fit(img1, img2, (x1,x2,y1,y2))

# original template
plt.imshow(T[x1:x2,y1:y2],cmap='gray')
plt.title('Template'),plt.axis('off'),plt.show()
# without Lucas Kanade
plt.imshow(img2[x1:x2,y1:y2],cmap='gray')
plt.title('No Lucas Kanade'),plt.axis('off'),plt.show()
# with Lucas Kanade
plt.imshow(IW,cmap='gray')
plt.title('Lucas Kanade'),plt.axis('off'),plt.show()