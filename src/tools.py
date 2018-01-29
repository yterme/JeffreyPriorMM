import numpy as np


def ratio(num, den, log=False) :
    log_nom = [np.log(num[i]) for i in range(len(num))]
    log_den = [np.log(den[i]) for i in range(len(den))]
    log_ratio = sum(log_nom) - sum(log_den)

    if not log :
        return np.exp(log_ratio)
    return log_ratio