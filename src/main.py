#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:51:33 2018

@author: yannickterme
"""

from functions import *
from JeffreyPrior import JeffreyPrior
import matplotlib.pyplot as plt
import pandas as pd
from BivariatePlotter import *
K = 2

###### Parameters ########
synthetic = True # synthetic or real data
# which parameters are unknown
unknown= [1,2] # set to [1] for mu unknown,  [0]: w unknown,[1, 2]: mu & sigma unknown etc.

###########################

def main(synthetic=True, unknown=[1]):

    
    if synthetic:
        mu_true = [1, -4]
        sigma_true = [0.2, 2]
        w_true = [0.3]  # w1 here
        
        n = 1000
        
        x1 = np.random.normal(mu_true[0], sigma_true[0], int(np.round(w_true[0] * n)))
        x2 = np.random.normal(mu_true[1], sigma_true[1], n - int(np.round(w_true[0] * n)))
        
        x = np.concatenate([x1, x2])
    else:
        w_true=[0.5]
        base = pd.read_csv('../height.txt')
        mu1=np.mean(base[base.Gender=="Male"].Height)
        mu2=np.mean(base[base.Gender=="Female"].Height)
        mu_true=[mu1,mu2]
        std1=np.std(base[base.Gender=="Male"].Height)
        std2=np.std(base[base.Gender=="Female"].Height)
        sigma_true=[std1,std2]
        heights=base.Height.values
        np.random.shuffle(heights)
        x=heights[:500].astype(float)
        #plt.hist(x, bins=10)
    # Delayed Acceptance Algo
    
    # Initialization
    if 0 in unknown:
        w=[0.5]
    else:
        w = w_true
    if 1 in unknown:
        mu=[np.mean(x), np.mean(x)+0.1]
    else:
        mu=mu_true
    mu = [0,0.1]
    if 2 in unknown:
        sigma=[np.std(x), np.std(x)+0.1]
    else:
        sigma=sigma_true
    
    k = GaussianKernel(std_w=0.1, std_mu=0.1, std_sigma=0.1)
    p = JeffreyPrior(riemann_parameters={"splits": 1000, "bounds": \
                                         [-2*abs(np.min(x)), 2*abs(np.max(x))]}, debug_mode=False)
    # If debug mode is True, Jeffrey returns always 0
    m = Mixture(x)
    trace_w = []
    trace_mu = []
    trace_sigma = []
    # Acceptance rate
    acc_list = []
    
    
    N = 5000
    trace_w=list(trace_w)
    trace_mu=list(trace_mu)
    trace_sigma=list(trace_sigma)
    
    for i in range(N):
        u1 = np.random.uniform()
        u2 = np.random.uniform()
        #kernel simulation
        (w_prop, mu_prop, sigma_prop) = k.simulate(w, mu, sigma)
        # 6 next lines are here to make the code flexible to known and unknown parameters
        if not(0 in unknown):
            w_prop=w
        if not(1 in unknown):
            mu_prop=mu
        if not(2 in unknown):
            sigma_prop=sigma
        # compute log-ratios
        diff_l = m.likelihood(w_prop, mu_prop, sigma_prop, log=True) - \
                 m.likelihood(w, mu, sigma, log=True)
        diff_k = k.evaluate(w, mu, sigma, w_prop, mu_prop, sigma_prop, log=True) - \
                 k.evaluate(w_prop, mu_prop, sigma_prop, w, mu, sigma, log=True)
        acc = 0 
        if np.log(u1) < diff_l + diff_k:
            p_prop=p.evaluate(w_prop, mu_prop, sigma_prop, proportional=False, density=m.density(w_prop, mu_prop, sigma_prop), log=True, unknown=unknown) 
            p0=p.evaluate(w, mu, sigma, proportional=False, density=m.density(w, mu, sigma), log=True, unknown=unknown)
            print("Log ratio of priors:", p_prop-p0)
            if np.log(u2) < p_prop-p0:
                w = w_prop
                mu = mu_prop
                sigma = sigma_prop
                acc = 1
        acc_list.append(acc)
    
        if i % 50 == 0:
            print("i=",i)
        #update traces
        trace_w.append(w)
        trace_mu.append(mu)
        trace_sigma.append(sigma)
    
    acc_rate = np.mean(acc_list)    
    
    trace_w = np.array(trace_w)
    trace_mu = np.array(trace_mu)
    trace_sigma = np.array(trace_sigma)
    
    burn=500
    
    
    plt.hist(trace_w)
    plt.hist(trace_mu[burn:, 0])
    plt.hist(trace_mu[burn:, 1])
    plt.hist(trace_sigma[:, 0], 10)
    plt.hist(trace_sigma[:, 1])
    
    
    
    #bivariate_plotter_kernel (63, 65, 68, 71, 0.01,trace_mu[burn:, 0] , trace_mu[burn:, 1], \
    #                          title = "", name = "real_mu.pdf")
    f=plt.figure()
    bivariate_plotter_kernel (min(trace_mu[burn:, 0]),max(trace_mu[burn:, 0]) , \
                              min(trace_mu[burn:, 1]),max(trace_mu[burn:, 1]), \
                              0.01, trace_mu[burn:, 0] , trace_mu[burn:, 1], \
                              )
    plt.xlabel("mu1")
    plt.ylabel("mu2")
    plt.show()
    f.savefig("synthetic_mu2.pdf", bbox_inches='tight')
    
    f=plt.figure()
    bivariate_plotter_kernel (min(trace_sigma[burn:, 0]),max(trace_sigma[burn:, 0]) , \
                              min(trace_sigma[burn:, 1]),max(trace_sigma[burn:, 1]), \
                              0.01, trace_sigma[burn:, 0] , trace_sigma[burn:, 1], \
                              )
    plt.xlabel("sigma1")
    plt.ylabel("sigma2")
    plt.show()
    f.savefig("synthetic_sigma2.pdf", bbox_inches='tight')
    
