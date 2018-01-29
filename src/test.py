from scipy.stats import norm
import numpy as np

from Integral import Integral

# Test the simple integral

riemann_parameters = {"splits": 1000,
                      "bounds": [-10, 10]}

density_function = lambda x: norm.pdf(x, loc=0, scale=1)
function = lambda x: 1

integral = Integral(riemann_parameters=riemann_parameters)

integral_value = integral.integrate(function=function, density=density_function)
print(integral_value)


def functions_mappings(i, j) :
    def fun(x) :
        return norm.pdf(x, loc=0, scale=1)
    return fun

# Test the matrix integral
functions_matrix = [[functions_mappings(i, j) for i in range(4)] for j in range(1, 5)]
integral_value = integral.integrate_matrix(functions_matrix=functions_matrix, density=density_function)
print(integral_value)