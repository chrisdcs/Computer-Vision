# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:15:25 2021

@author: m1390
"""


import numpy as np
import scipy.signal as ss
import scipy.stats as st
# %%
class Lucas_Kanade:

    def __init__(self, I, max_iter, tol):

        # I is the initialization image, assume image size don't change
        self.transform = Transform2D(I)
        
        # initialize warping
        # affine transform, can be generalized by projective transform
        self.W = np.column_stack([np.eye(2), np.zeros((2, 1))])

        self.p = np.zeros(6)

        # current image
        self.I = I

        # tolerance/threshold for stopping criterion
        self.max_iter = max_iter

        # dx and dy are kernels for computing image gradients
        self.d_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.d_y = self.d_x.T
        
        # tolerance for least square template registration stopping criterion
        self.tol = tol
        
        self.history = []

    def fit(self, T, I, coordinates):

        x1, x2, y1, y2 = coordinates

        iteration = 0

        while True:
            # warp image using warping matrix W (step 1)
            IW = self.transform.fit(x1, x2, y1, y2, self.W, I)

            # compute the error image (step 2)
            error_img = T[x1:x2,y1:y2] - IW

            # compute and warp gradient ∇I with W(x,p) (step 3)
            Ix = ss.convolve2d(I, self.d_x, 'same')
            Iy = ss.convolve2d(I, self.d_y, 'same')

            # warp gradient
            IWx = self.transform.fit(x1, x2, y1, y2, self.W, Ix)
            IWy = self.transform.fit(x1, x2, y1, y2, self.W, Iy)
            gradI = np.zeros((2, IWx.shape[0]*IWx.shape[1]))
            gradI[0] = IWx.flatten()
            gradI[1] = IWy.flatten()
            gradI = gradI.T

            # compute Jacobian (step 4) and Steepest Descent images (step 5)
            Jacobian = self.transform.compute_jacobian(x1, x2, y1, y2, self.W)
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
                break
        
        IW = self.transform.fit(x1, x2, y1, y2, self.W, I)
        return IW

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

        return IW

    def fit(self, x1, x2, y1, y2, W, img):

        # generate normalized grid
        h = x2 - x1
        w = y2 - y1
        x = np.linspace(-1, 1, h+1)[:-1]
        y = np.linspace(-1, 1, w+1)[:-1]

        # create homogeneous coordinates
        xt, yt = np.meshgrid(x, y)
        ones = np.ones(np.prod(xt.shape))
        grid = np.vstack([xt.flatten(), yt.flatten(), ones])

        grid_transform = W @ grid
        grid_transform = grid_transform.reshape(-1, h, w, order='F')

        xt1 = grid_transform[0]
        yt1 = grid_transform[1]

        # transform back to the range of (x1,x2), (y1,y2)
        x_new = (xt1 + 1) / 2 * h + x1
        y_new = (yt1 + 1) / 2 * w + y1

        IW = self.interpolation(x_new, y_new, img)

        return IW

    def compute_jacobian(self, x1, x2, y1, y2, W):

        # compute Jacobian matrix of W evaluated at x,y
        h = x2 - x1
        w = y2 - y1
        x = np.linspace(-1, 1, h+1)[:-1]
        y = np.linspace(-1, 1, w+1)[:-1]

        Jacobian = np.zeros((h*w, 2, 6))
        idx = 0
        for i in range(h):
            for j in range(w):
                J = np.array([[x[i], 0, y[j], 0, 1, 0],
                             [0, x[i], 0, y[j], 0, 1]])
                Jacobian[idx] = J
                idx += 1

        return Jacobian


def normalize_img(img):
    return (img - np.min(img))/255


class Corner:

    def __init__(self, I):

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

    def compute_gradient(self, I):

        # Sobel operator for computing image gradient
        d_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        d_y = d_x.T

        # I = cv2.imread('box.jpg',0)
        self.Ix = ss.convolve2d(I, d_x, 'same')
        self.Iy = ss.convolve2d(I, d_y, 'same')

    def compute_corner(self):

        Ix = self.Ix
        Iy = self.Iy

        Ixy = Ix * Iy

        Ixx = ss.convolve2d(Ix**2, self.gkern(5, 1), 'same')
        Iyy = ss.convolve2d(Iy**2, self.gkern(5, 1), 'same')
        Ixy = ss.convolve2d(Ixy, self.gkern(5, 1), 'same')

        det = Ixx * Iyy - Ixy ** 2
        trace = Ixx + Iyy

        harris = det - 0.06*trace**2

        harris = (harris - harris.mean())/harris.std()
        self.corners = (harris > .3)
