from Integral import Integral
from numpy.linalg import det

from double_derivatives import *


class JeffreyPrior():
    def __init__(self, mcmc_parameters=None, riemann_parameters=None):
        self.functions_dict = {(0,0) : d2_w2,
                               (0,1) : d2_wi_muj,
                               (0,2) : d2_wi_sigmaj,
                               (1,0) : d2_wi_muj,
                               (1,1) : d2_mui_muj,
                               (1,2) : d2_mui_sigmaj,
                               (2,0) : d2_wi_sigmaj,
                               (2,1) : d2_mu_i_muj,
                               (2,2) : d2_sigmai_sigmaj}

        self.integral = Integral(riemann_parameters=riemann_parameters, mcmc_parameters=mcmc_parameters)

    def functions_matrix(self, w, mu, sigma):
        matrix_size = len(w) + len(mu) + len(sigma)
        belongings_row = [0 for i in range(len(w))] + [1 for i in range(len(mu))] + [2 for i in range(len(sigma))]
        belongings_matrix = [[(belongings_row[i], belongings_row[j]) for i in range(matrix_size)]
                            for j in range(matrix_size)]
        return [[self.functions_dict[belongings_matrix[i, j]] for i in range(matrix_size)]
                            for j in range(matrix_size)]

    def information_matrix(self, w, mu, sigma, density):
        """ Information matrix with Rieman integral"""
        functions_matrix = self.functions_matrix(w, mu, sigma)
        return self.integral.integrate_matrix(functions_matrix, density=density)

    def evaluate(self, w, mu, sigma, density, log):
        if log :
            return np.log(det(self.information_matrix(w, mu, sigma, density)))
        else :
            return det(self.information_matrix(w, mu, sigma, density))
