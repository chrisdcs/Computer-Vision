# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:14:49 2020

@author: Chris Ding
"""
import cv2
import numpy as np
import os

from utils.MACE import MACE
from utils.tools import plot_correlation_plane
#%% Load Data
trainD = np.zeros((10,64,64)).astype(np.uint8)
# print(os.getcwd())

for i in range(10):
    fname = r'faceExpressionDatabase\A'+ str(i // 10) + str(i % 10) + '.bmp'
    trainD[i] = cv2.imread(fname,-1)
    
testD = np.zeros((60,64,64)).astype(np.uint8)
for i in range(60):
    fname = r'faceExpressionDatabase\D' + str(i // 10) + str(i % 10) + '.bmp'
    testD[i] = cv2.imread(fname,-1)    
#%%
maceA = MACE(trainD)
maceA.train()

maceB = MACE(testD)
maceB.train()
#%% plot things

Corr,peak = maceA.test(testD[4],compute_Corr = True)

plot_correlation_plane(Corr,title_name='MACE Filter False')


Corr1,peak1 = maceB.test(testD[4],compute_Corr = True)

# Plot the surface.
plot_correlation_plane(Corr1,title_name='MACE Filter True')

print("False image center correlation:\n",peak)
print("True image center correlation:\n",peak1)