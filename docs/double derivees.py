# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:10:35 2018

@author: Utilisateur
"""

import sympy
import scipy
import numpy as np
import math
import scipy.stats

scipy.stats.norm.pdf(1, [0], [2])

dnorm = scipy.stats.norm.pdf


def d2_sigma2 (i, p, mu, sigma, x, proportionnal = True) : 
    if proportionnal : 
        return((p[i]**2/sigma[i]**2)*
           (((x-mu[i])**2/sigma[i]**2 -1 )*dnorm(x, mu[i], sigma[i]))**2
           /np.dot(p,dnorm(x,mu, sigma)))
    return((p[i]**2/sigma[i]**2)*
           (((x-mu[i])**2/sigma[i]**2 -1 )*dnorm(x, mu[i], sigma[i]))**2
           /np.dot(p,dnorm(x,mu, sigma)))

d2_sigma2(0, [.5, .5], [0,1], [1, 1], 0  )
d2_sigma2(1, [.5, .5], [0,1], [1, 1], 0  )



def d2_sigmai_sigmaj (i,j, p, mu, sigma, x) : 
    return(((p[i]*p[j])/(sigma[i]*sigma[j]))*
           (((x-mu[i])**2/sigma[i]**2 -1 )*dnorm(x, mu[i], sigma[i]))*
           (((x-mu[j])**2/sigma[j]**2 -1 )*dnorm(x, mu[j], sigma[j]))
           /np.dot(p,dnorm(x,mu, sigma)))
    

#on trouve pareil #schwarz  
d2_sigmai_sigmaj(0,1, [.5, .5], [0,1], [1, 1], 0  )
d2_sigmai_sigmaj(1,0, [.5, .5], [0,1], [1, 1], 0  )
    
d2_sigmai_sigmaj(0,1, [.5, .5], [0,1], [1, 1], 10  )
d2_sigmai_sigmaj(1,0, [.5, .5], [0,1], [1, 1], 10  )


def d2_mui_sigmai (i, p, mu, sigma, x) : 
    return((p[i]**2/sigma[i]**3)*
           (x-mu[i])*((x-mu[i])**2/sigma[i]**2 -1 )*dnorm(x, mu[i], sigma[i])**2
           /np.dot(p,dnorm(x,mu, sigma)))
    
#on trouve 0 #logique
d2_mui_sigmai(0, [.5, .5], [0,1], [1, 1], 0  )
d2_mui_sigmai(1, [.5, .5], [0,1], [1, 1], 0  )    

d2_mui_sigmai(1, [.5, .5], [0,1], [1, 1], 10  )    

    
def d2_mui_sigmaj (i,j, p, mu, sigma, x) : 
    return(((p[i]*p[j])/(sigma[i]*sigma[j]))*
           ((x-mu[i])/(sigma[i]^2*sigma[j]))*
           (((x-mu[j])**2/sigma[j]**2 -1 )*
            dnorm(x, mu[j], sigma[j])*dnorm(x, mu[i], sigma[i]))
           /np.dot(p,dnorm(x,mu, sigma))) 
           
           
d2_mui_sigmaj(0,1, [.5, .5], [0,1], [1, 1], 0  )


def d2_pi_pj (i,j, p, mu, sigma, x) : 
    return(-dnorm(x, mu[i], sigma[i])*dnorm(x, mu[j], sigma[j])/(np.dot(p,dnorm(x,mu, sigma))**2))
    
#on trouve pareil #schwarz
d2_pi_pj(0,1, [.5, .5], [0,1], [1, 1], 0  )
d2_pi_pj(1,0, [.5, .5], [0,1], [1, 1], 0  )

def d2_pi2 (i, p, mu, sigma, x) : 
    return(-dnorm(x, mu[i], sigma[i])**2/(np.dot(p,dnorm(x,mu, sigma))**2))
   
d2_pi2(1, [.5, .5], [0,1], [1, 1], 0  )


def d2_pi_muj (i,j, p, mu, sigma, x) : 
    return(p[j]*((mu[j]-x)/sigma[j]**2)*dnorm(x, mu[j], sigma[j])*dnorm(x, mu[i], sigma[i])/(np.dot(p,dnorm(x,mu, sigma))**2))
    
d2_pi_muj(0,1, [.5, .5], [0,1], [1, 1], 0  )


def d2_pi_mui (i, p, mu, sigma, x) : 
    return (((mu[i]-x)/sigma[i]**2)*(dnorm(x, mu[i], sigma[i])/(np.dot(p,dnorm(x,mu, sigma)))-p[i]*(dnorm(x, mu[i], sigma[i])/(np.dot(p,dnorm(x,mu, sigma))))**2))

d2_pi_mui(0, [.5, .5], [0,1], [1, 1], 0  )



def d2_pi_sigmaj (i,j, p, mu, sigma, x) : 
    return(p[j]/sigma[j]*((x-mu[j])**2/sigma[j]**2-1)*dnorm(x, mu[i], sigma[i])*dnorm(x, mu[j], sigma[j])/(np.dot(p,dnorm(x,mu, sigma))**2))
    
d2_pi_sigmaj(0,1, [.5, .5], [0,1], [1, 1], 0  )

def d2_pi_sigmai (i, p, mu, sigma, x) : 
    return(1/sigma[i]*((mu[i]-x)**2/sigma[i]**2-1)*(p[i]*dnorm(x, mu[i], sigma[i])/np.dot(p,dnorm(x,mu, sigma))-dnorm(x, mu[i], sigma[i])**2/np.dot(p,dnorm(x,mu, sigma))**2))
    
d2_pi_sigmai(0, [.5, .5], [0,1], [1, 1], 0  )

def d2_mui2 (i, p, mu, sigma, x) : 
    return(p[i]/sigma[i]**2*dnorm(x, mu[i], sigma[i])/np.dot(p,dnorm(x,mu, sigma))*(1-(x-mu[i])**2/sigma[i]**2)*(1-p[i]*dnorm(x, mu[i], sigma[i])/np.dot(p,dnorm(x,mu, sigma))))
    