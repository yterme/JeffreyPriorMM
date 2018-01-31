from Integral import Integral

from second_derivatives import SecondDerivatives

from numpy.linalg import det
import numpy as np


class JeffreyPrior():
    def __init__(self, mcmc_parameters=None, riemann_parameters=None, debug_mode=False):
        self.integral = Integral(riemann_parameters=riemann_parameters, mcmc_parameters=mcmc_parameters)
        self.second_derivatives = SecondDerivatives()
        self.debug_mode = debug_mode

    def functions_matrix(self, w, mu, sigma, proportional, unknown):
        """ Returns the matrix of second derivatives functions for computing the information matrix"""
        

        matrix_size = 0
        vars_idx = []
        belongings_row = []

        parameters = [w, mu, sigma]
        
        for par_idx in unknown :
            matrix_size+= len(parameters[par_idx])
            vars_idx.append([i for i in range(len(parameters[par_idx]))])
            belongings_row.append([par_idx for _ in range(len(parameters[par_idx]))])

        belongings_row = [x for sublist in belongings_row for x in sublist]
        vars_idx = [x for sublist in vars_idx for x in sublist]

        #print(belongings_row)
        #print(vars_idx)
        belongings_matrix = np.array([[(belongings_row[i], belongings_row[j])
                                       for i in range(matrix_size)]
                                      for j in range(matrix_size)])
        #print(belongings_matrix)
        #import pdb; pdb.set_trace()
        #print("Idx",vars_idx)
        return([[self.second_derivatives.functions_mappings(couple=belongings_matrix[i, j],
                                         w=w, mu=mu, sigma=sigma, i=vars_idx[i], j=vars_idx[j],
                                                            proportional=proportional)
                 for i in range(matrix_size)] for j in range(matrix_size)])

    def information_matrix(self, w, mu, sigma, proportional, density, known):
        """ Information matrix with Riemann integral"""
        functions_matrix = self.functions_matrix(w, mu, sigma, proportional, known)
        mat =  self.integral.integrate_matrix(functions_matrix, density=density)
        #print(mat)
        return(mat)

    def evaluate(self, w, mu, sigma, proportional, density, log, unknown):
        if self.debug_mode :
            return 0

        #print(self.information_matrix(w, mu, sigma, proportional, density, known))
        if log:
            return 0.5 * np.log(det(-np.array(self.information_matrix(w, mu, sigma, proportional, density, unknown))))
        else:
            return np.sqrt(det(-np.array(self.information_matrix(w, mu, sigma, proportional, density, unknown))))