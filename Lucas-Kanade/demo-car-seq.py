# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:06:02 2021

@author: m1390
"""

# import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.LucasKanade import Lucas_Kanade
from utils.miscellaneous import normalize_img

# for car sequence it's better to use a tight bounding box
frames = np.load('data/carseq.npy')

box = (116, 151, 59, 145)
box0 = (116, 151, 59, 145)

coords = None
rectList = []
time_total = 0
seq_len = frames.shape[2]

max_iter = 150
tol = 0.05

LKT = Lucas_Kanade(frames[:,:,0], max_iter, tol, box)
frame0 = frames[:,:,0]

# W_list = []

for i in range(1,seq_len):
    
    print("Processing frame %d" % i)
    T = frames[:,:,i-1]
    I = frames[:,:,i]
    
    box, IW = LKT.fit(T, I, box)
    
    width = box[3] - box[2]
    height = box[1] - box[0]
    
    
    box0 = (box0[0],box0[0]+height,box0[2],box0[2]+width)
    if i % 5 == 0 or ((i>= 270) and (i <= 290)) or (i>= 310):
        # This is a template correction step
        box, Iw = LKT.fit(frame0, I, box, box0)
    
    # if i % 50 == 0:
    #     frame0 = I.copy()
    #     box0 = box
    
    
    if i % 10 == 0 or i == 1:
        plt.figure()
        plt.imshow(frames[:,:,i],cmap='gray')
        bbox = patches.Rectangle((int(box[2]), int(box[0])), width, height,
                                 fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(bbox)
        plt.title('frame %d'%i)
        plt.show()