#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:34:21 2018

@author: yannickterme
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import truncnorm

#class Kernel:
#    def __init__(self, which="gaussian"):

        
class GaussianKernel():
    def __init__(self, size_w,std_mu, std_sigma):
        self.size_w=size_w
        self.std_mu= std_mu
        self.std_sigma=std_sigma
        
    def simulate(self, w, mu, sigma):
            #print("test")
            return (list(np.random.uniform(max(0,w[0]-self.size_w/2),min(1, w[0]+self.size_w/2),len(w))),\
                    list(np.random.normal(mu,[self.std_mu,self.std_mu])), \
                    [truncnorm.rvs(-sigma[0]/self.std_sigma, sigma[0]+10*self.std_sigma, loc=sigma[0], scale=self.std_sigma), \
                    truncnorm.rvs(-sigma[1]/self.std_sigma, sigma[1]+10*self.std_sigma, loc=sigma[1], scale=self.std_sigma)] )                
                    #list(np.exp(np.random.normal([np.log(s) for s in sigma], [1,1]))) )
                    #np.random.lognormal(, [1,1]))   

    def evaluate(self, w, mu , sigma, w_ctr, mu_ctr, sigma_ctr):
        return (1 * rbf_kernel(mu[0], mu_ctr[0])[0][0] * rbf_kernel(mu[1], mu_ctr[1])[0][0]* \
                    truncnorm.pdf(x=sigma[0], a=-sigma_ctr[0]/self.std_sigma,b= sigma_ctr[0]+10*self.std_sigma, loc=sigma_ctr[0], scale=self.std_sigma)* \
                    truncnorm.pdf(x=sigma[1],a=-sigma_ctr[1]/self.std_sigma, b=sigma_ctr[1]+10*self.std_sigma, loc=sigma_ctr[1], scale=self.std_sigma))                  
               
                #rbf_kernel(np.log(sigma[0]), np.log(sigma_ctr[0]))[0][0] *\
                #rbf_kernel(np.log(sigma[1]), np.log(sigma_ctr[1]))[0][0])
    
        
class Mixture():
    def __init__(self,x):
        self.x=x
    def likelihood(self, w, mu, sigma):
        w_all=w+[1-sum(w)]
        sigma2=[s**2 for s in sigma ]
        #assumption gaussian mixture
        return(sum([w_all[i]*(-0.5*np.log(2*np.pi*sigma2[i])- abs((xj-mu[i])**2/sigma2[i])) \
                    for i in range(len(mu)) for xj in self.x]))
        
class JeffreyPrior():
    def __init__(self):
        self.type="jeffrey"
    def evaluate(self, w, mu, sigma):
        # for test
        return 1
    
