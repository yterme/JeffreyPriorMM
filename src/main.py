#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:51:33 2018

@author: yannickterme
"""

from priors import *
from functions import write_to_json
from tqdm import tqdm


K = 2

# Synthetic data parameters
mu_true = [1, -4]
sigma_true = [0.2, 2]
w_true = [0.3, 0.7]  # w1 here

# Sample the data
n = 100

x1 = np.random.normal(mu_true[0], sigma_true[0], int(np.round(w_true[0] * n)))
x2 = np.random.normal(mu_true[1], sigma_true[1], n - int(np.round(w_true[0] * n)))

x = np.concatenate([x1, x2])

# Delayed Acceptance Algorithm

# Initialization
w = [0.5, 0.5]
mu = [0, -4]
sigma = [0.2, 2]
nb_sim = 1000

std_w = 1
std_mu = 1
std_sigma = 0.1


# Objects initialization

k = GaussianKernel(std_w=std_w, std_mu=std_mu, std_sigma=std_sigma)
p = JeffreyPrior()
m = Mixture(x)

trace_w = []
trace_mu = []
trace_sigma = []
trace_acc = []
diffs_1 = []
diffs_2 = []

# MH algorithm

for i in tqdm(range(nb_sim)):
    acc = 0
    u1 = np.random.uniform()
    u2 = np.random.uniform()
    (w_prop, mu_prop, sigma_prop) = k.simulate(w, mu, sigma)

    log_likelihood_diff = m.likelihood(w_prop, mu_prop, sigma_prop, "log") - m.likelihood(w, mu, sigma, "log")
    kernel_diff = k.evaluate(w, mu, sigma, w_prop, mu_prop, sigma_prop, "log") - \
                  k.evaluate(w_prop, mu_prop, sigma_prop, w, mu, sigma, "log")
    diff_1 = log_likelihood_diff + kernel_diff
    diff_2 = np.nan

    if np.log(u1) < diff_1:
        diff_2 = p.evaluate(w_prop, mu_prop, sigma_prop) / p.evaluate(w, mu, sigma)
        if np.log(u2) < diff_2:
            w = w_prop
            mu = mu_prop
            sigma = sigma_prop
            acc = 1
    print(w)
    print("mu" + str(mu))
    trace_w.append(w)
    trace_mu.append(mu)
    trace_sigma.append(sigma)
    diffs_1.append(diff_1)
    diffs_2.append(diff_2)
    trace_acc.append(acc)

write_to_json(trace_w, trace_mu, trace_sigma, trace_acc, diffs_1, diffs_2)