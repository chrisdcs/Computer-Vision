# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:29:34 2021

@author: Ding Chi
"""

import numpy as np
import scipy.signal as ss
import scipy.stats as st

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