from scipy import integrate

class Integral():
    def __init__(self, riemann_parameters=None, mcmc_parameters=None):

        if mcmc_parameters is not None :
            self.type = "mcmc"

        if riemann_parameters is not None :
            self.type = "riemann"
            self.splits = riemann_parameters["splits"]
            self.bounds = riemann_parameters["bounds"]
            self.low_bound = self.bounds[0]
            self.up_bound = self.bounds[1]
            self.riemann_interval = (self.up_bound - self.low_bound) / self.splits

    def integrate(self, function, density):
        #evaluation_points = [self.low_bound + i * self.riemann_interval for i in range(self.splits)]
        #import pdb; pdb.set_trace()
        #values = [function(evaluation_points[i]) * density(evaluation_points[i]) for i in range(self.splits)]
        def fun_to_int(x):
            return(function(x)*density(x))
        return(integrate.quad(fun_to_int, self.low_bound, self.up_bound)[0])
        
        #return sum(values) * self.riemann_interval

    def integrate_matrix(self, functions_matrix, density):
        #matrix_size = len(functions_matrix)
        return [[self.integrate(function= function, density=density)
                 for function in function_list]
                for function_list in functions_matrix]
