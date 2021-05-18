# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:32:39 2021

@author: Ding Chi
"""

import numpy as np

def normalize_img(img):
    return (img - np.min(img))/255