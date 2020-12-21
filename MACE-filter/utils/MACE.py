# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 10:00:39 2020

@author: m1390
"""
import numpy as np

class MACE:
    """
        MACE filter class:
            Minimum Average Correlation Energy filter implementation
            
            X: D x N frequency domain data matrix (fft)
            D: average spectral energy matrix (diagnal)
            H: frequency MACE filter
            
            trainD: a matrix of image training data with size N x H X W
                H: height
                W: weight
                
            H, W are supposed to be the same (square matrix for images)
            
        Note: input training data are supposed to be normalized between 0-255,
        if not, better set regularize as True.
    """
    
    def __init__(self,trainD,regularize=False):
        
        self.trainD = trainD
        
        self.n = trainD.shape[0]
        
        self.height = trainD.shape[1]
        
        self.width = trainD.shape[2]
        
        self.X = None
        
        self.Xcon = None
        
        self.D = None
        
        self.Dinv = None
        
        self.H = None
        
        self.correlation_filter = None
        
        self.regularize = regularize
        
    def train(self):
        
        
        for i in range(self.n):
            
            
            dft = np.fft.fft2(self.trainD[i])
            # fftshit for spatial correlation filter visualization
            dft = np.fft.fftshift(dft)
            dft = dft.flatten()

            if self.X is None:
                self.X = dft
            else:
                self.X = np.c_[self.X, dft.flatten()]
        
        """
            + 10 is a term for minimizing noise variance, 
            but it will give a less sharp peak
            refer paper:
                Tutorial survey of composite filter designs for optical correlators, 
                by B. V. K. Vijaya Kumar
        """
        self.Dinv = 1./(np.mean(np.abs(self.X)**2,1)+10)
        
        self.Xcon = self.X.conj().T
        
        c = np.ones((self.n,1))
        
        # regularization for numerical instability
        if self.regularize:
            self.H = (self.Dinv.reshape(-1,1) * self.X) @ \
                     np.linalg.inv(self.Xcon * self.Dinv @ self.X \
                     + 0.01 * np.eye(self.n)) @ c
        else:
            self.H = (self.Dinv.reshape(-1,1) * self.X) @ \
                     np.linalg.inv(self.Xcon * self.Dinv @ self.X ) @ c
        
        # inverse fft MACE filter to understand what features are learnt 
        # in spatial domain
        self.correlation_filter = np.abs(np.fft.ifft2(np.fft.ifftshift(self.H.reshape(self.height,self.width))))
        
    def test(self,image,compute_Corr=False):
        
        # fourier transform
        imagefft = np.fft.fftshift(np.fft.fft2(image))
        
        peak = np.abs(imagefft.flatten().conj()@self.H.flatten())
        if compute_Corr is False:
            return peak
        else:
            Corr_Freq = imagefft * np.conjugate(self.H).reshape(self.height,self.width)
            Corr = np.abs(np.fft.ifft2(Corr_Freq))
            Corr = np.fft.fftshift(Corr)
            
            Corr = Corr * peak/Corr[self.height//2,self.width//2]
            
            return Corr,peak