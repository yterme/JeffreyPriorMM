import numpy as np
import scipy.stats

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
        def fun(x):
            return self.functions_dict[tuple(couple)](i=i, j=j, p=w, mu=mu, sigma=sigma, x=x, proportional=proportional)
        return fun

    @staticmethod
    def d2_pi_pj(i, j, p, mu, sigma, x, proportional=None):
        if i == j:
            return -dnorm(x, mu[i], sigma[i]) ** 2 / (np.dot(p, dnorm(x, mu, sigma)) ** 2)
        return -dnorm(x, mu[i], sigma[i]) * dnorm(x, mu[j], sigma[j]) / (np.dot(p, dnorm(x, mu, sigma)) ** 2)

    @staticmethod
    def d2_pi_muj(i, j, p, mu, sigma, x, proportional=None):
        if i == j:
            return ((mu[i] - x) / sigma[i] ** 2) * (dnorm(x, mu[i], sigma[i]) / (np.dot(p, dnorm(x, mu, sigma))) - p[i] * (
                    dnorm(x, mu[i], sigma[i]) / (np.dot(p, dnorm(x, mu, sigma)))) ** 2)
        return (p[j] * ((mu[j] - x) / sigma[j] ** 2) * dnorm(x, mu[j], sigma[j]) * dnorm(x, mu[i], sigma[i]) / (
                np.dot(p, dnorm(x, mu, sigma)) ** 2))

    @staticmethod
    def d2_pi_sigmaj(i, j, p, mu, sigma, x, proportional=None):
        if i == j:
            return 1 / sigma[i] * ((mu[i] - x) ** 2 / sigma[i] ** 2 - 1) * (
                    p[i] * dnorm(x, mu[i], sigma[i]) / np.dot(p, dnorm(x, mu, sigma)) - dnorm(x, mu[i], sigma[i]) ** 2 /
                    np.dot(p, dnorm(x, mu, sigma)) ** 2)
        return (p[j] / sigma[j] * ((x - mu[j]) ** 2 / sigma[j] ** 2 - 1) * dnorm(x, mu[i], sigma[i]) *
                dnorm(x, mu[j], sigma[j]) / (np.dot(p, dnorm(x, mu, sigma)) ** 2))

    @staticmethod
    def d2_mui_muj(i, j, p, mu, sigma, x, proportional=None):
        if i == j:
            return (p[i] / sigma[i] ** 2 * dnorm(x, mu[i], sigma[i]) / np.dot(p, dnorm(x, mu, sigma)) *
                    (1 - (x - mu[i]) ** 2 / sigma[i] ** 2) * (1 - p[i] * dnorm(x, mu[i], sigma[i]) /
                                                              np.dot(p, dnorm(x, mu, sigma))))
        else:
            # TODO : Check this derivative
            return - p[i] * p[j] * ((x - mu[i]) / sigma[i] ** 2) * ((x - mu[j]) / sigma[j] ** 2) * \
                   (dnorm(x, mu[i], sigma[i]) * dnorm(x, mu[j], sigma[j])) / np.dot(p, dnorm(x, mu, sigma))

    @staticmethod
    def d2_mui_sigmaj(i, j, p, mu, sigma, x, proportional=None):
        if i == j:
            return ((p[i] ** 2 / sigma[i] ** 3) *
                    (x - mu[i]) * ((x - mu[i]) ** 2 / sigma[i] ** 2 - 1) * dnorm(x, mu[i], sigma[i]) ** 2
                    / np.dot(p, dnorm(x, mu, sigma)))
        return (((p[i] * p[j]) / (sigma[i] * sigma[j])) * ((x - mu[i]) / (sigma[i] ** 2 * sigma[j])) *
                (((x - mu[j]) ** 2 / sigma[j] ** 2 - 1) * dnorm(x, mu[j], sigma[j]) * dnorm(x, mu[i], sigma[i]))
                / np.dot(p, dnorm(x, mu, sigma)))

    @staticmethod
    def d2_sigmai_sigmaj(i, j, p, mu, sigma, x, proportional=None):
        if i == j:
            if proportional:
                return ((p[i] ** 2 / sigma[i] ** 2) *
                        (((x - mu[i]) ** 2 / sigma[i] ** 2 - 1) * dnorm(x, mu[i], sigma[i])) ** 2
                        / np.dot(p, dnorm(x, mu, sigma)))

            return ((p[i] ** 2 / sigma[i] ** 2) *
                    (((x - mu[i]) ** 2 / sigma[i] ** 2 - 1) * dnorm(x, mu[i], sigma[i])) ** 2
                    / np.dot(p, dnorm(x, mu, sigma)))
        return (((p[i] * p[j]) / (sigma[i] * sigma[j])) *
                (((x - mu[i]) ** 2 / sigma[i] ** 2 - 1) * dnorm(x, mu[i], sigma[i])) *
                (((x - mu[j]) ** 2 / sigma[j] ** 2 - 1) * dnorm(x, mu[j], sigma[j]))
                / np.dot(p, dnorm(x, mu, sigma)))
