# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:31:10 2021

@author: Ding Chi
"""

import numpy as np
from scipy import signal
from utils.Transform2D import Transform2D

class Lucas_Kanade:

    def __init__(self, I, max_iter, tol, box):

        # I is the initialization image, assume image size don't change
        self.transform = Transform2D(I)
        
        # initialize warping
        # affine transform, can be generalized by projective transform
        self.W = np.column_stack([np.eye(2), np.zeros((2, 1))])
        self.W_old = None

        self.p = np.zeros(6)

        # current image
        self.I = I

        # tolerance/threshold for stopping criterion
        self.max_iter = max_iter
        
        # tolerance for least square template registration stopping criterion
        self.tol = tol
        
        self.history = []
        
        self.h = box[1] - box[0]
        self.w = box[3] - box[2]

    def fit(self, T, I, box, box0 = None):

        x1, x2, y1, y2 = box

        iteration = 0

        while True:
            # warp image using warping matrix W (step 1)
            _,IW = self.transform.fit(box, self.W, I)

            # compute the error image (step 2)
            if box0 is None:
                template = T[x1:x2,y1:y2]
                error_img = template - IW
            else:
                x10,x20,y10,y20 = box0
                template = T[x10:x20,y10:y20]
                error_img = template - IW

            # compute and warp gradient ∇I with W(x,p) (step 3)
            Ix, Iy = np.gradient(I)

            # warp gradient
            _,IWx = self.transform.fit(box, self.W, Ix)
            _,IWy = self.transform.fit(box, self.W, Iy)
            
            gradI = np.zeros((2, IWx.shape[0]*IWx.shape[1]))
            gradI[0] = IWx.flatten()
            gradI[1] = IWy.flatten()
            gradI = gradI.T

            # compute Jacobian (step 4) and Steepest Descent images (step 5)
            Jacobian = self.transform.compute_jacobian(box, self.W)
            SDI = (gradI[:, None, :] @ Jacobian).squeeze()

            # compute Hessian and solve least squares
            # H = SDI.T @ SDI
            # delta_p = np.linalg.inv(H) @ (error.flatten() @ SDI)

            # numerically more stable using QR for Least Squares
            Q, R = np.linalg.qr(SDI)
            delta = np.linalg.inv(R) @ Q.T @ error_img.flatten()
            
            # print(np.linalg.norm(delta))
            self.p = self.p + delta
            self.history.append(delta)
            
            # update parameters
            self.W[0,0] = 1 + self.p[0]
            self.W[1,0] = self.p[1]
            self.W[0,1] = self.p[2]
            self.W[1,1] = 1 + self.p[3]
            self.W[0,2] = self.p[4]
            self.W[1,2] = self.p[5]

            if np.linalg.norm(delta) < self.tol:
                break

            iteration += 1
            if iteration >= self.max_iter:
                # if it does not converge, use the previous parameters
                self.W = self.W_old.copy()
                break
        
        box, IW = self.transform.fit(box, self.W, I)
        
        # Current W becomes old W
        self.W_old = self.W.copy()
        
        # tight bounding box setting
        box = (box[0],np.clip(box[1],box[0]+self.h,box[0]+self.h),
               box[2],np.clip(box[3],box[2]+self.w,box[2]+self.w))
        
        return box, IW, np.corrcoef(template.reshape(-1), IW.reshape(-1))[0][1]