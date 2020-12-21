# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 10:23:42 2020

@author: m1390
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def plot_correlation_plane(plane,title_name='Title None'):
    """
    

    Parameters
    ----------
    plane : numpy array.
        size x size correlation plane.
    title_name : string, optional
        title name for displaying in the figure. The default is 'Title None'.

    Returns
    -------
    None.

    """
    size,size = plane.shape
    
    X = np.linspace(-1, 1, size) * size//2
    Y = np.linspace(-1, 1, size) * size//2
    X, Y = np.meshgrid(X, Y)
    
    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, plane, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # Customize the z axis.
    ax.set_zlim(0, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_zticks(np.arange(0,1.1,0.2))
    ax.set_title(title_name)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()
    
def vote(classification):
    """
    This function do a max-voting procedure to determine which object is being viewed.

    Parameters
    ----------
    classification : numpy array
        Classification decisions for each step.

    Returns
    -------
    final : N x 1 array
        Final decisions for N objects that are being viewed.

    """
    final = np.zeros((16))
    
    n_obj,steps = classification.shape
    
    vote_matrix = np.zeros((16,n_obj))
    
    for i in range(n_obj):
        
        for j in range(steps):
            
            decision = int(classification[i,j])
        
            vote_matrix[decision,i] += 1
        
    final = np.argmax(vote_matrix,0)
    
    return final

def train(trainData,testData,MACE_LIST):
    """
    This function train a set of MACE filters on training data

    Parameters
    ----------
    trainData : numpy data matrix
        size N x width x height.
    testData : TYPE
        DESCRIPTION.
    MACE_LIST : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    n = len(MACE_LIST)
    