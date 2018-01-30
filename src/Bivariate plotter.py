# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:28:13 2018

@author: Seb
"""

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def bivariate_plotter (x_min, x_max, y_min, y_max, delta, f, title = "") : 
    x = np.arange(x_min, x_max, delta)
    y = np.arange(y_min, y_max, delta)
    X, Y = np.meshgrid(x, y)
    Z =f(X, Y)
    plt.figure()
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
def f(x, y) : 
    return(mlab.bivariate_normal(x, y, 1.0, 1.0, 3,3))
bivariate_plotter(-5, 5, -5, 5, 0.1, f)

def f(x, y) : 
    return(x*y)
bivariate_plotter(-5, 5, -5, 5, 0.1, f)