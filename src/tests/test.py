from scipy.stats import norm
import numpy as np

from double_derivatives import *
from Integral import Integral

# Test the simple integral

riemann_parameters = {"splits": 1000,
                      "bounds": [-10, 10]}

density_function = lambda x: norm.pdf(x, loc=0, scale=1)
function = lambda x: 1

integral = Integral(riemann_parameters=riemann_parameters)

integral_value = integral.integrate(function=function, density=density_function)
print(integral_value)


# Test the matrix integral
functions_matrix = [[functions_mappings(i, j) for i in range(4)] for j in range(1, 5)]
integral_value = integral.integrate_matrix(functions_matrix=functions_matrix, density=function)
print(integral_value)

## Double derivation tests


# on trouve 0 #logique
d2_mui_sigmai(0, [.5, .5], [0, 1], [1, 1], 0)
d2_mui_sigmai(1, [.5, .5], [0, 1], [1, 1], 0)

d2_mui_sigmai(1, [.5, .5], [0, 1], [1, 1], 10)





d2_mui_sigmaj(0, 1, [.5, .5], [0, 1], [1, 1], 0)


d2_sigma2(0, [.5, .5], [0, 1], [1, 1], 0)
d2_sigma2(1, [.5, .5], [0, 1], [1, 1], 0)

# on trouve pareil #schwarz
d2_sigmai_sigmaj(0, 1, [.5, .5], [0, 1], [1, 1], 0)
d2_sigmai_sigmaj(1, 0, [.5, .5], [0, 1], [1, 1], 0)

d2_sigmai_sigmaj(0, 1, [.5, .5], [0, 1], [1, 1], 10)
d2_sigmai_sigmaj(1, 0, [.5, .5], [0, 1], [1, 1], 10)




# on trouve pareil #schwarz
d2_pi_pj(0, 1, [.5, .5], [0, 1], [1, 1], 0)
d2_pi_pj(1, 0, [.5, .5], [0, 1], [1, 1], 0)





d2_pi2(1, [.5, .5], [0, 1], [1, 1], 0)





d2_pi_muj(0, 1, [.5, .5], [0, 1], [1, 1], 0)





d2_pi_mui(0, [.5, .5], [0, 1], [1, 1], 0)





d2_pi_sigmaj(0, 1, [.5, .5], [0, 1], [1, 1], 0)




d2_pi_sigmai(0, [.5, .5], [0, 1], [1, 1], 0)



