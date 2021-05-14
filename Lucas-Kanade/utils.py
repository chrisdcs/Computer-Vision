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
    
    def __init__(self,I):
        self.W = None
        self.I = I
        
        self.height = I.shape[0]
        self.width = I.shape[1]
        
    def interpolation(self):
        pass
        
    def fit(self,x1,x2,y1,y2,W,img):
        
        # generate normalized grid
        h = x2 - x1
        w = y2 - y1
        x = np.linspace(-1, 1, h+1)[:-1]
        y = np.linspace(-1, 1, w+1)[:-1]
        
        # create homogeneous coordinates
        xt,yt = np.meshgrid(x,y)
        ones = np.ones(np.prod(xt.shape))
        grid = np.vstack([xt.flatten(),yt.flatten(),ones])
        
        grid_transform = W @ grid
        grid_transform = grid_transform.reshape(-1,h,w,order='F')
        
        xt1 = grid_transform[0]
        yt1 = grid_transform[1]
        
        # transform back to the range of (x1,x2), (y1,y2)
        x_new = (xt1 + 1) / 2 * h + x1
        y_new = (yt1 + 1) / 2 * w + y1
        
        x0 = np.floor(x_new).astype(np.int64)
        x1 = x0 + 1
        y0 = np.floor(y_new).astype(np.int64)
        y1 = y0 + 1
        
        x0 = np.clip(x0,0,self.height-1)
        y0 = np.clip(y0,0,self.width-1)
        x1 = np.clip(x1,0,self.height-1)
        y1 = np.clip(y1,0,self.width-1)

        Ia = img[x0,y0]
        Ib = img[x0,y1]
        Ic = img[x1,y0]
        Id = img[x1,y1]
        
        # linear interpolation coefficients
        wa = (x1-x_new) * (y1-y_new)
        wb = (x1-x_new) * (y_new-y0)
        wc = (x_new-x0) * (y1-y_new)
        wd = (x_new-x0) * (y_new-y0)
        
        # add dimension for addition
        if len(img.shape) > 2:
            wa = np.expand_dims(wa, axis=2)
            wb = np.expand_dims(wb, axis=2)
            wc = np.expand_dims(wc, axis=2)
            wd = np.expand_dims(wd, axis=2)
        
        # compute output
        IW = wa*Ia + wb*Ib + wc*Ic + wd*Id
        
        return IW#Ia
    
    def compute_jacobian(self,x1,x2,y1,y2,W):
        
        # compute Jacobian matrix of W evaluated at x,y
        h = x2 - x1
        w = y2 - y1
        x = np.linspace(x1, x2, h+1)[:-1]
        y = np.linspace(y1, y2, w+1)[:-1]
        
        Jacobian = np.zeros((h*w,2,6))
        idx = 0
        for i in range(h):
            for j in range(w):
                J = np.array([[x[i],0,y[j],0,1,0],[0,x[i],0,y[j],0,1]])
                Jacobian[idx] = J
                idx += 1
        
        return Jacobian
        

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