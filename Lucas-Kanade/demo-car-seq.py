# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:06:02 2021

@author: m1390
"""

# import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.LucasKanade import Lucas_Kanade

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

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('results/result.mp4', fourcc, 30.0, (320,  240))

for i in range(1,seq_len):
    
    print("Processing frame %d" % i)
    T = frames[:,:,i-1]
    I = frames[:,:,i]
    
    box, IW = LKT.fit(T, I, box)
    
    width = box[3] - box[2]
    height = box[1] - box[0]
    
    
    box0 = (box0[0],box0[0]+height,box0[2],box0[2]+width)
    
    if i % 5 == 0 or ((i>= 270) and (i <= 290)) or (i>= 310):
        # If statement is for speed up computation
        
        # This is a template correction step
        box, Iw = LKT.fit(frame0, I, box, box0)
    
    # add bounding box to each frame
    I = ((I-I.min())/(I.max()-I.min())*255).astype(np.uint8)
    I = cv2.cvtColor(I, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(I, (box[2], box[0]), (box[3], box[1]), (0, 0, 255), 2)
    
    # write frames to video
    out.write(I)
    
    # plot every 10 frames
    if i % 50 == 0 or i == 1:
        plt.figure()
        plt.imshow(I)
        plt.title('frame %d'%i)
        plt.show()
        
out.release()