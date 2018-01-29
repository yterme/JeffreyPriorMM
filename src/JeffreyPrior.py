from Integral import Integral

from second_derivatives import SecondDerivatives

from numpy.linalg import det
import numpy as np


class JeffreyPrior():
    def __init__(self, mcmc_parameters=None, riemann_parameters=None, debug_mode=False):
        self.integral = Integral(riemann_parameters=riemann_parameters, mcmc_parameters=mcmc_parameters)
        self.second_derivatives = SecondDerivatives()
        self.debug_mode = debug_mode

    def functions_matrix(self, w, mu, sigma, proportional):
        """ Returns the matrix of second derivatives functions for computing the information matrix"""

        matrix_size = len(w) + len(mu) + len(sigma)
        vars_idx = [i for i in range(len(w))] + [i for i in range(len(mu))] + [i for i in range(len(sigma))]

        belongings_row = [0 for _ in range(len(w))] + [1 for _ in range(len(mu))] + [2 for _ in range(len(sigma))]
        belongings_matrix = np.array([[(belongings_row[i], belongings_row[j])
                                       for i in range(matrix_size)]
                                      for j in range(matrix_size)])

        return [[self.second_derivatives.functions_mappings(couple=belongings_matrix[i, j],
                                         w=w, mu=mu, sigma=sigma, i=vars_idx[i], j=vars_idx[j],
                                                            proportional=proportional)
                 for i in range(matrix_size)] for j in range(matrix_size)]

    def information_matrix(self, w, mu, sigma, proportional, density):
        """ Information matrix with Riemann integral"""
        functions_matrix = self.functions_matrix(w, mu, sigma, proportional)
        mat= -self.integral.integrate_matrix(functions_matrix, density=density)
        print(mat)
        return(mat)

    def evaluate(self, w, mu, sigma, proportional, density, log):
        if self.debug_mode :
            return 0

        # TODO : Check the consistency of the dimension of w throughout the code, in this class w is the full w
        #assert .99 < sum(w) < 1.01
        if log:
            return 0.5*np.log(det(self.information_matrix(w, mu, sigma, proportional, density)))
        else:
            return np.sqrt(det(self.information_matrix(w, mu, sigma, proportional, density)))