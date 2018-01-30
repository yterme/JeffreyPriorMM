import numpy as np
import scipy.stats
from tools import ratio

dnorm = scipy.stats.norm.pdf


class SecondDerivatives():
    def __init__(self):
        self.functions_dict = {(0, 0): self.d2_pi_pj,
                               (0, 1): self.d2_pi_muj,
                               (0, 2): self.d2_pi_sigmaj,
                               (1, 0): self.d2_pi_muj,
                               (1, 1): self.d2_mui_muj,
                               (1, 2): self.d2_mui_sigmaj,
                               (2, 0): self.d2_pi_sigmaj,
                               (2, 1): self.d2_mui_sigmaj,
                               (2, 2): self.d2_sigmai_sigmaj}

    def functions_mappings(self, couple, w, mu, sigma, i, j, proportional):
        p=w+[1-sum(w)]
        def fun(x):
            #print(self.functions_dict[tuple(couple)])
            #print(couple)
            return self.functions_dict[tuple(couple)](i=i, j=j, p=p, mu=mu, sigma=sigma, x=x, proportional=proportional)

        return fun

    def mixture_density(self, p, mu, sigma):
        return

    @staticmethod
    def d2_pi_pj(i, j, p, mu, sigma, x, proportional=None):
        num = [- dnorm(x, mu[i], sigma[i]), dnorm(x, mu[j], sigma[j])]
        den = [np.dot(p, dnorm(x, mu, sigma)) ** 2]
        return ratio(num=num, den=den)

    @staticmethod
    def d2_pi_muj(i, j, p, mu, sigma, x, proportional=None):
        if i == j:
            num = [(x - mu[i]), dnorm(x, mu[i], sigma[i]),
                   (p[i] * dnorm(x, mu[i], sigma[i]) - np.dot(p, dnorm(x, mu, sigma)))]
            den = [sigma[i] ** 2, np.dot(p, dnorm(x, mu, sigma)) ** 2]
        else:
            num = [p[j], (mu[j] - x), dnorm(x, mu[j], sigma[j]), dnorm(x, mu[i], sigma[i])]
            den = [sigma[j] ** 2, np.dot(p, dnorm(x, mu, sigma)) ** 2]
        return ratio(num=num, den=den)

    @staticmethod
    def d2_pi_sigmaj(i, j, p, mu, sigma, x, proportional=None):
        if i == j:
            diff = ratio(num=[np.dot(p, dnorm(x, mu, sigma)) * dnorm(x, mu[i], sigma[i]) -
                              p[i] * dnorm(x, mu[i], sigma[i] ** 2)],
                         den=[np.dot(p, dnorm(x, mu, sigma)) ** 2])
            num = [(mu[i] - x) ** 2 / (sigma[i] ** 2) - 1, diff]
            den = [sigma[i]]
        else:
            num = [- p[j], (x - mu[j]) ** 2 / sigma[j] ** 2 - 1, dnorm(x, mu[i], sigma[i]), dnorm(x, mu[j], sigma[j])]
            den = [sigma[j], (np.dot(p, dnorm(x, mu, sigma)) ** 2)]
        return ratio(num, den)

    @staticmethod
    def d2_mui_muj(i, j, p, mu, sigma, x, proportional=None):
        if i == j:
            diff = 1 - ratio([p[i] * dnorm(x, mu[i], sigma[i])], [np.dot(p, dnorm(x, mu, sigma))])
            num = [p[i], dnorm(x, mu[i], sigma[i]), (1 - (x - mu[i]) ** 2 / sigma[i] ** 2), diff]
            den = [sigma[i] ** 2, np.dot(p, dnorm(x, mu, sigma))]
        else:
            # TODO : Check this derivative
            num = [- p[i], p[j], x - mu[i], x - mu[j], dnorm(x, mu[i], sigma[i]), dnorm(x, mu[j], sigma[j])]
            den = [sigma[i] ** 2, sigma[j] ** 2, np.dot(p, dnorm(x, mu, sigma)) ** 2]
        return ratio(num, den)

    @staticmethod
    def d2_mui_sigmaj(i, j, p, mu, sigma, x, proportional=None):
        if i == j:
            num = [p[i] ** 2, (x - mu[i]), ((x - mu[i]) ** 2 / sigma[i] ** 2 - 1), dnorm(x, mu[i], sigma[i]) ** 2]
            den = [sigma[i] ** 3, np.dot(p, dnorm(x, mu, sigma))]
        else:
            num = [(p[i] * p[j]), (x - mu[i]), ((x - mu[j]) ** 2 / sigma[j] ** 2 - 1), dnorm(x, mu[j], sigma[j]),
                   dnorm(x, mu[i], sigma[i])]
            den = [(sigma[i] * sigma[j]), (sigma[i] ** 2 * sigma[j]), np.dot(p, dnorm(x, mu, sigma))]
        return ratio(num, den)

    @staticmethod
    def d2_sigmai_sigmaj(i, j, p, mu, sigma, x, proportional=None):
        # if proportional:
        #     num = [(p[i] ** 2 / sigma[i] ** 2), (((x - mu[i]) ** 2 / sigma[i] ** 2 - 1) * dnorm(x, mu[i], sigma[i])) ** 2]
        #     den = [np.dot(p, dnorm(x, mu, sigma))]
        num = [(p[i] * p[j]), ((x - mu[i]) ** 2 / sigma[i] ** 2 - 1), dnorm(x, mu[i], sigma[i]),
               ((x - mu[j]) ** 2 / sigma[j] ** 2 - 1), dnorm(x, mu[j], sigma[j])]
        den = [(sigma[i] * sigma[j]), np.dot(p, dnorm(x, mu, sigma))]
        return ratio(num, den)
