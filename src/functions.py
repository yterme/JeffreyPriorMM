#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:34:21 2018

@author: yannickterme
"""

from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import truncnorm
import numpy as np

class GaussianKernel():
    def __init__(self, std_w, std_mu, std_sigma, sigma_kernel="TruncNorm"):
        # self.size_w=size_w`
        self.std_w = std_w
        self.std_mu = std_mu
        self.std_sigma = std_sigma
        self.sigma_kernel = sigma_kernel

    def simulate(self, w, mu, sigma):
        return (self.simulate_w(w), self.simulate_mu(mu), self.simulate_sigma(sigma))

    def simulate_w(self, w):
        return (list([truncnorm.rvs(a=-w[0] / self.std_w, b=(1 - w[0]) / self.std_w, loc=w[0], scale=self.std_w)]))

    def simulate_mu(self, mu):
        return (list(np.random.normal(mu, [self.std_mu, self.std_mu])))


    def simulate_sigma(self, sigma):
        if self.sigma_kernel == "TruncNorm":
            return ([truncnorm.rvs(-sigma[0] / self.std_sigma, sigma[0] + 10 * self.std_sigma, loc=sigma[0],
                                   scale=self.std_sigma), \
                     truncnorm.rvs(-sigma[1] / self.std_sigma, sigma[1] + 10 * self.std_sigma, loc=sigma[1],
                                   scale=self.std_sigma)])
        elif self.sigma_kernel == "LogNorm":
            return ([lognorm.rvs(s=self.std_sigma, loc=sigma[0], scale=np.exp(sigma[0])), \
                     lognorm.rvs(s=self.std_sigma, loc=sigma[1], scale=np.exp(sigma[1]))])

    def evaluate(self, w, mu, sigma, w_ctr, mu_ctr, sigma_ctr, log=False):
        if log:
            return (self.evaluate_w(w, w_ctr, True) + self.evaluate_mu(mu, mu_ctr, True) + self.evaluate_sigma(sigma,
                                                                                                               sigma_ctr,
                                                                                                               True))
        else:
            return (self.evaluate_w(w, w_ctr, False) * self.evaluate_mu(mu, mu_ctr, False) * self.evaluate_sigma(sigma,
                                                                                                                 sigma_ctr,
                                                                                                                 False))

    def evaluate_w(self, w, w_ctr, log=False):
        if not (log):
            return (truncnorm.pdf(x=w[0], a=-w_ctr[0] / self.std_w, \
                                  b=(1 - w_ctr[0]) / self.std_w, loc=w_ctr[0], scale=self.std_w))
        else:
            return (truncnorm.logpdf(x=w[0], a=-w_ctr[0] / self.std_w, \
                                     b=(1 - w_ctr[0]) / self.std_w, loc=w_ctr[0], scale=self.std_w))

    def evaluate_mu(self, mu, mu_ctr, log=False):
        if not (log):
            return (norm.pdf(x=mu[0], loc=mu_ctr[0], scale=self.std_mu) * \
                    norm.pdf(x=mu[1], loc=mu_ctr[1], scale=self.std_mu))
        else:
            return (norm.logpdf(x=mu[0], loc=mu_ctr[0], scale=self.std_mu) + \
                    norm.pdf(x=mu[1], loc=mu_ctr[1], scale=self.std_mu))

    def evaluate_sigma(self, sigma, sigma_ctr, log=False):
        if self.sigma_kernel == "TruncNorm":
            if not (log):
                return (truncnorm.pdf(x=sigma[0], a=-sigma_ctr[0] / self.std_sigma,
                                      b=sigma_ctr[0] + 10 * self.std_sigma, loc=sigma_ctr[0], scale=self.std_sigma) + \
                        truncnorm.pdf(x=sigma[1], a=-sigma_ctr[1] / self.std_sigma,
                                      b=sigma_ctr[1] + 10 * self.std_sigma, loc=sigma_ctr[1], scale=self.std_sigma))

            else:
                return (truncnorm.logpdf(x=sigma[0], a=-sigma_ctr[0] / self.std_sigma,
                                         b=sigma_ctr[0] + 10 * self.std_sigma, loc=sigma_ctr[0], scale=self.std_sigma) * \
                        truncnorm.logpdf(x=sigma[1], a=-sigma_ctr[1] / self.std_sigma,
                                         b=sigma_ctr[1] + 10 * self.std_sigma, loc=sigma_ctr[1], scale=self.std_sigma))
        elif self.sigma_kernel == "LogNorm":
            if not (log):
                return (lognorm.pdf(s=self.std_sigma, x=sigma[0], loc=sigma_ctr[0], scale=self.std_sigma) * \
                        lognorm.pdf(s=self.std_sigma, x=sigma[1], loc=sigma_ctr[1], scale=self.std_sigma))
            else:
                return (lognorm.logpdf(s=self.std_sigma, x=sigma[0], loc=sigma_ctr[0], scale=self.std_sigma) + \
                        +lognorm.logpdf(s=self.std_sigma, x=sigma[1], loc=sigma_ctr[1], scale=self.std_sigma))


class Mixture():
    def __init__(self, x):
        self.x = x

    def likelihood(self, w, mu, sigma, log=True):

        w_all=w+[1-sum(w)]
        # sigma2=[s**2 for s in sigma ]
        # assumption gaussian mixture
        ls = [sum([w_all[i] * norm.pdf(xj, loc=mu[i], scale=sigma[i]) \
                   for i in range(len(w_all))]) for xj in self.x]
        return np.sum([np.log(l) for l in ls])

    @staticmethod
    def density(w, mu, sigma):
        w_all=w+[1-sum(w)]
        def density_fun(x):
            return sum([w_all[i] * norm.pdf(x, loc=mu[i], scale=sigma[i]) for i in range(len(w_all))])
        return density_fun
