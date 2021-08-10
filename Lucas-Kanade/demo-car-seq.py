# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:06:02 2021

@author: Chi, Ding
"""

# import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.LucasKanade import Lucas_Kanade

# for car sequence it's better to use a tight bounding box
frames = np.load('data/carseq.npy')

box = (116+5, 151-5, 59+5, 145-5)
box0 = (116+5, 151-5, 59+5, 145-5)

coords = None
rectList = []
time_total = 0
seq_len = frames.shape[2]

max_iter = 150
tol = 0.01

record = False

LKT = Lucas_Kanade(frames[:,:,0], max_iter, tol, box)
frame0 = frames[:,:,0]

if record:
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('results/result.mp4', fourcc, 30.0, (320,  240))

for i in range(1,seq_len):
    
    
    T = frames[:,:,i-1]
    I = frames[:,:,i]
    
    box, IW, corr = LKT.fit(T, I, box)
    print("Processing frame %d" % i, corr)
    
    width = box[3] - box[2]
    height = box[1] - box[0]
        
    # This is a template correction step
    box, Iw, _ = LKT.fit(frame0, I, box, box0)
    
    # add bounding box to each frame
    I = ((I-I.min())/(I.max()-I.min())*255).astype(np.uint8)
    I = cv2.cvtColor(I, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(I, (box[2], box[0]), (box[3], box[1]), (0, 0, 255), 2)
    
    # write frames to video
    if record:
        out.write(I)
    
    # plot every 10 frames
    if i % 50 == 0 or i == 0:
        plt.figure()
        plt.imshow(I)
        plt.title('frame %d'%i)
        plt.show()
    
    # update template once in a while
    if corr > 0.99 and i % 100 == 0:
        frame0 = frames[:,:,i]
        box0 = box

if record:
    out.release()