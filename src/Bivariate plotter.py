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




import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy
import sklearn
from sklearn.neighbors.kde import KernelDensity
from scipy import stats
import matplotlib.pyplot as plt

def bivariate_plotter_kernel (x_min, x_max, y_min, y_max, delta, x, y, title = "", name = "out.pdf") : 
    data = np.vstack([x,y])
    x = np.arange(x_min, x_max, delta)
    y = np.arange(y_min, y_max, delta)
    X, Y = np.meshgrid(x, y)
    data  = data
    kde = stats.gaussian_kde(data)
    print(kde(np.array([1,5])))
    Z =[]
    for i in range(len(X)) : 
        Z.append(kde(np.vstack([X[i],Y[i]])))
    Z = np.array(Z)
    print(Z)
    
    f = plt.figure()
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)    
    f.savefig(name, bbox_inches='tight')
bivariate_plotter_kernel(-5, 5, -5, 5, 0.1, np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000), name = "\\paradis\eleves\SCOUBE\Bureau\out.pdf")
