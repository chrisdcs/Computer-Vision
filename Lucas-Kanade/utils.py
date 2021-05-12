# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:15:25 2021

@author: m1390
"""

import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.stats as st
#%%
def warp(h,w,W):
    # h is height
    # w is width
    # W is the warping matrix
    x = np.linspace(-1, 1, h)
    y = np.linspace(-1, 1, w)
    
    
    return

class Transform2D:
    """
        This class computes the affine 2D transform
            To Do: Should add projective transform as well
    """
    
    def __init__(self,):
        self.W = None
        
    def interpolation(self):
        pass
        
    def fit(self,x1,x2,y1,y2,W):
        # generate normalized grid
        h = x2 - x1 + 1
        w = y2 - y1 + 1
        x = np.linspace(-1, 1, h)
        y = np.linspace(-1, 1, w)
        
        # create homogeneous coordinates
        xt,yt = np.meshgrid(x,y)
        ones = np.ones(np.prod(xt.shape))
        grid = np.vstack([xt.flatten(),yt.flatten(),ones])
        
        grid_transform = W @ grid
        grid_transform = grid_transform.reshape(-1,h,w,order='F')
        
        xt1 = grid_transform[0]
        yt1 = grid_transform[1]
        
        # transform back to the range of (x1,x2), (y1,y2)
        x_new = (xt1 * h + x1 * 2) / 2
        y_new = (yt1 * w + y1 * 2) / 2

def normalize_img(img):
    return (img - np.min(img))/255

class Corner:
    
    def __init__(self,I):
        
        self.Ix = None
        self.Iy = None
        self.I = I
        
        self.corners = None
    

    def gkern(self, kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel."""
    
        x = np.linspace(-nsig, nsig, kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d/kern2d.sum()
    
    def compute_gradient(self,I):
    
        # Sobel operator for computing image gradient
        d_x = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        d_y = d_x.T
    
        # I = cv2.imread('box.jpg',0)
        self.Ix = ss.convolve2d(I,d_x,'same')
        self.Iy = ss.convolve2d(I,d_y,'same')
    
    def compute_corner(self):
        
        Ix = self.Ix
        Iy = self.Iy
        
        Ixy = Ix * Iy
        
        Ixx = ss.convolve2d(Ix**2,self.gkern(5,1),'same')
        Iyy = ss.convolve2d(Iy**2,self.gkern(5,1),'same')
        Ixy = ss.convolve2d(Ixy,self.gkern(5,1),'same')

        det = Ixx * Iyy - Ixy ** 2
        trace = Ixx + Iyy

        harris = det - 0.06*trace**2
        
        harris = (harris - harris.mean())/harris.std()
        self.corners = (harris > .3)