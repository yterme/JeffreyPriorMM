#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:51:33 2018

@author: yannickterme
"""

from functions import *
from JeffreyPrior import JeffreyPrior
import matplotlib.pyplot as plt

K = 2
# Synthetic data parameters
mu_true = [1, -4]
sigma_true = [0.2, 2]
w_true = [0.3]  # w1 here

n = 1000

x1 = np.random.normal(mu_true[0], sigma_true[0], int(np.round(w_true[0] * n)))
x2 = np.random.normal(mu_true[1], sigma_true[1], n - int(np.round(w_true[0] * n)))

x = np.concatenate([x1, x2])

# Delayed Acceptance Algo

# Initialization
w = [0.5]
mu = [0, 0]
sigma = [1, 1]
N = 1000

k = GaussianKernel(std_w=0.1, std_mu=0.1, std_sigma=0.1)
p = JeffreyPrior(riemann_parameters={"splits": 1000, "bounds": [-30, 30]}, debug_mode=False)
# If debug mode is True, Jeffrey returns always 0
m = Mixture(x)
trace_w = []
trace_mu = []
trace_sigma = []
# Acceptance rate
acc_list = []

for i in range(N):
    u1 = np.random.uniform()
    u2 = np.random.uniform()
    (w_prop, mu_prop, sigma_prop) = k.simulate(w, mu, sigma)
    diff_l = m.likelihood(w_prop, mu_prop, sigma_prop) - \
             m.likelihood(w, mu, sigma)
    diff_k = k.evaluate(w, mu, sigma, w_prop, mu_prop, sigma_prop, log=True) - \
             k.evaluate(w_prop, mu_prop, sigma_prop, w, mu, sigma, log=True)
    acc = 0
    if np.log(u1) < diff_l + diff_k:
        # TODO : Generalize full w throughout the code
        # !! : Risky code chirurgy
        # Here we need the full omegas
        w_full = w + [1 - sum(w)]
        w_prop_full = w_prop + [1 - sum(w_prop)]
        if (np.log(u2) < p.evaluate(w_prop_full, mu_prop, sigma_prop, proportional=False, density=m.density(w_full, mu, sigma), log=True) -
                p.evaluate(w_full, mu, sigma, proportional=False, density=m.density(w_full, mu, sigma), log=True)):
            w = w_prop
            mu = mu_prop
            sigma = sigma_prop
            acc = 1
    acc_list.append(acc)

    # print(acc)
    if i % 50 == 0:
        print(i)
    trace_w.append(w)
    trace_mu.append(mu)
    trace_sigma.append(sigma)

acc_rate = np.mean(acc_list)

trace_w = np.array(trace_w)
trace_mu = np.array(trace_mu)
trace_sigma = np.array(trace_sigma)

plt.hist(trace_w)
plt.hist(trace_mu[:, 0])
plt.hist(trace_mu[:, 1])
plt.hist(trace_sigma[:, 0])
plt.hist(trace_sigma[:, 1])
