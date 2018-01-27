import numpy as np
from scipy.stats import truncnorm

import json
from time import time
from time import strftime


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    """A simplified parametrization of the scipy truncated normal function"""
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def product(input):
    """ Returns the product of a list while avoiding the underflow"""
    log_input = [np.log(x) for x in input]
    return np.exp(sum(log_input))


def write_to_json(trace_w, trace_mu, trace_sigma, trace_acc, diffs_1, diffs_2):
    """ Writes the simulation data to a json file in the logs directory"""
    simulation_dict = {"w": trace_w,
                       "mu": trace_mu,
                       "sigma": trace_sigma,
                       "acc": trace_acc,
                       "diffs_1": diffs_1,
                       "diffs_2": diffs_2}

    now = strftime("%c")
    log_dir = "logs/" + now.format(time())
    with open(log_dir + '-simulation-data.json', 'w+') as json_file:
        json.dump(simulation_dict, json_file, sort_keys=True, indent=4)