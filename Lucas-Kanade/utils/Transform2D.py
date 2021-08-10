# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:30:34 2021

@author: Ding Chi
"""

import numpy as np


class Transform2D:
    """
        This class computes the affine 2D transform
            To Do: Should add projective transform as well
    """

    def __init__(self, I):
        self.W = None
        self.I = I

        self.height = I.shape[0]
        self.width = I.shape[1]

    def interpolation(self, x, y, img):
        # only linear for now

        # x,y are double float type coordinates
        x0 = np.floor(x).astype(np.int64)
        x1 = x0 + 1
        y0 = np.floor(y).astype(np.int64)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, self.height-1)
        y0 = np.clip(y0, 0, self.width-1)
        x1 = np.clip(x1, 0, self.height-1)
        y1 = np.clip(y1, 0, self.width-1)

        Ia = img[x0, y0]
        Ib = img[x0, y1]
        Ic = img[x1, y0]
        Id = img[x1, y1]

        # linear interpolation coefficients
        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        # add dimension for addition
        if len(img.shape) > 2:
            wa = np.expand_dims(wa, axis=2)
            wb = np.expand_dims(wb, axis=2)
            wc = np.expand_dims(wc, axis=2)
            wd = np.expand_dims(wd, axis=2)

        # compute output
        IW = wa*Ia + wb*Ib + wc*Ic + wd*Id

        return (x0.min(),x0.max()+1,y0.min(),y0.max()+1), IW

    def fit(self, box, W, img):
        
        x1,x2,y1,y2 = box
        
        # generate grid
        h = x2 - x1
        w = y2 - y1

        x = np.linspace(x1, x2, h+1)[:-1]
        y = np.linspace(y1, y2, w+1)[:-1]

        # create homogeneous coordinates
        xt, yt = np.meshgrid(x, y)
        ones = np.ones(np.prod(xt.shape))
        grid = np.vstack([xt.flatten(), yt.flatten(), ones])

        grid_transform = W @ grid
        grid_transform = grid_transform.reshape(-1, h, w, order='F')

        x_new = grid_transform[0]
        y_new = grid_transform[1]
        
        """
        # An alternative
        
        grid = np.array([[x1,x2],[y1,y2],[1,1]])
        grid_transform = W @ grid
        
        x1_new = grid_transform[0,0]
        x2_new = grid_transform[0,1]
        y1_new = grid_transform[1,0]
        y2_new = grid_transform[1,1]
        
        x = np.linspace(x1_new, x2_new, int(x2 - x1) + 1)[:-1]
        y = np.linspace(y1_new, y2_new, int(y2 - y1) + 1)[:-1]
        
        y_new, x_new = np.meshgrid(y,x)
        """
        
        box,IW = self.interpolation(x_new, y_new, img)

        return box,IW

    def compute_jacobian(self, box, W):
        
        x1,x2,y1,y2 = box
        # compute Jacobian matrix of W evaluated at x,y
        h = x2 - x1
        w = y2 - y1
        x = np.linspace(x1, x2, h+1)[:-1]
        y = np.linspace(y1, y2, w+1)[:-1]

        Jacobian = np.zeros((h*w, 2, 6))
        idx = 0
        for i in range(h):
            for j in range(w):
                J = np.array([[x[i], 0, y[j], 0, 1, 0],
                             [0, x[i], 0, y[j], 0, 1]])
                Jacobian[idx] = J
                idx += 1

        return Jacobian