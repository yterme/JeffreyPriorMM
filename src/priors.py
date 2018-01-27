#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:34:21 2018

@author: yannickterme
"""

import numpy as np
from scipy.stats import norm, dirichlet, lognorm
from functions import product

# class Kernel:
#    def __init__(self, which="gaussian"):


class GaussianKernel():
    def __init__(self, std_w, std_mu, std_sigma):
        self.std_w = std_w
        self.std_mu = std_mu
        self.std_sigma = std_sigma

    def simulate(self, w, mu, sigma):
        w_sample = list(np.random.dirichlet(self.std_w * np.array(w), 1)[0])
        sigma_sample = [np.random.lognormal(sigma[i], self.std_sigma, 1)[0] for i in range(len(mu))]
        mu_sample = [np.random.normal(mu[i], self.std_mu) for i in range(len(mu))]

        assert 0.999 < sum(w_sample) < 1.001

        return w_sample, mu_sample, sigma_sample

    @staticmethod
    def evaluate(w, mu, sigma, w_ctr, mu_ctr, sigma_ctr, type=None):
        """ evaluate w, mu, sigma | w_ctr, mu_ctr, sigma_ctr"""
        if type is None:
            w_eval = dirichlet.pdf(w, alpha=w_ctr)
            mu_evals = norm.pdf(mu, loc=mu_ctr, scale=sigma_ctr)
            sigma_evals = [lognorm.pdf(sigma[i], s=sigma_ctr[i]) for i in range(len(sigma))]
            sigma_eval = product(sigma_evals)
            mu_eval = product(mu_evals)
            return product([w_eval, mu_eval, sigma_eval])

        if type == "log":
            w_eval = dirichlet.logpdf(w, alpha=w_ctr)
            mu_evals = norm.logpdf(mu, loc=mu_ctr, scale=sigma_ctr)
            sigma_evals = [lognorm.logpdf(sigma[i], s=sigma_ctr[i]) for i in range(len(sigma))]

            mu_eval = sum(mu_evals)
            sigma_eval = sum(sigma_evals)
            return w_eval + mu_eval + sigma_eval


class Mixture():
    def __init__(self, x):
        self.x = x

    def likelihood(self, w, mu, sigma, type=None):
        likelihoods = [sum([w[i] * norm.pdf(xj, loc=mu[i], scale=sigma[i]) for i in range(len(w))]) for xj in self.x]
        if type is None :
            return product(likelihoods)
        if type == "log" :
            return sum([np.log(l) for l in likelihoods])


class JeffreyPrior():
    def __init__(self):
        self.type = "jeffrey"

    def evaluate(self, w, mu, sigma):
        # for test
        return 1

