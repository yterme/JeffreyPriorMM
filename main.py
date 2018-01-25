#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:51:33 2018

@author: yannickterme
"""

import numpy as np
from functions import *
import matplotlib.pyplot as plt
K=2
# Synthetic data parameters
mu_true=[1, -4]
sigma_true=[0.2, 2]
w_true=[0.3] #w1 here

n=100


x1=np.random.normal(mu_true[0],sigma_true[0],int(np.round(w_true[0]*n)))
x2=np.random.normal(mu_true[1],sigma_true[1],n-int(np.round(w_true[0]*n)))

x=np.concatenate([x1,x2])


# Delayed Acceptance Algo

#Initialization
w=[0.5]
mu=[0,0]
sigma= [1,1]
N=10000


k=GaussianKernel(size_w =0.5 ,std_mu=0.2, std_sigma=0.2)
p=JeffreyPrior()
m=Mixture(x)
trace_w=[]
trace_mu=[]
trace_sigma=[]
#Acceptance rate
acc=0
for i in range(N):
    u1=np.random.uniform()
    u2=np.random.uniform()
    (w_prop,mu_prop, sigma_prop) = k.simulate(w,mu, sigma)
    if u1< m.likelihood(w_prop,mu_prop, sigma_prop)  /  \
        m.likelihood(w,mu, sigma) * k.evaluate(w_prop,mu_prop, sigma_prop, w,mu,sigma) / \
        k.evaluate(w,mu, sigma, w_prop, mu_prop, sigma_prop):
        if u2<p.evaluate(w_prop,mu_prop, sigma_prop)/p.evaluate(w,mu, sigma):
            w=w_prop
            mu=mu_prop
            sigma=sigma_prop
            acc+=1
    trace_w.append(w)
    trace_mu.append(mu)
    trace_sigma.append(sigma)
    
acc=acc/N

trace_w=np.array(trace_w)
trace_mu=np.array(trace_mu)
trace_sigma=np.array(trace_sigma)

plt.hist(trace_w)
plt.hist(trace_mu[0,:])