########################################
############## EXAMPLES ################
######################################## 

# Dependencies:

import numpy as np 

# We setup a seed value for consistent randoms for testing. Using this seed allows us to choose a random non zero 'a' for linear functions.

np.random.seed(0)

'''
Important Notes:
    In this file we "hard code" the functions in, and we define the first and second derivative of each function.
    For some these examples we simply use wolfram alpha to save on time instead of doing it by hand.
    If the project structure was defined differently, I would have implemented a gradient function and a hessian approxiation function.
'''

# EXAMPLE 1:

def quadratic_1(x, hessian=False):
    Q = np.array([[1, 0], [0, 1]])
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if hessian else None
    return f, g, h

# Example 2:

def quadratic_2(x, hessian=False):
    Q = np.array([[1, 0], [0, 100]])
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if hessian else None
    return f, g, h

# Example 3:

def quadratic_3(x, hessian=False):
    r = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    Q = r.T @ np.diag([100, 1]) @ r
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if hessian else None
    return f, g, h

# Example 4:

def Rosenbrock(x, hessian=False):
    x1, x2 = x[0], x[1] # Splitting the values just for easier readability.
    f = 100*(x2 - x1**2)**2 + (1 - x1)**2
    g = np.array([ -400*x1*(x2 - x1**2) - 2*(1 - x1), 200*(x2 - x1**2) ])
    h = np.array([ [1200*x1**2 - 400*x2 + 2, -400*x1],[-400*x1, 200] ]) if hessian else None
    return f,g,h

# Example 5:

def linear_function(x, hessian=False):
    n = x.shape[0] # Get the dimension of the input vector.
    a = np.random.rand(n) # Note that for testing purposes, the vector "a" is reproducable since we defined a seed at the start of this program.
    f = a.T @ x
    g = a 
    h = np.zeros((n,n)) if hessian else None
    return f,g,h

# Example 6:

def exponential_function(x, hessian=False):
    x1 = x[0]
    x2 = x[1]
    e1 = np.exp(x1 + 3*x2 - 0.1)
    e2 = np.exp(x1 - 3*x2 - 0.1)
    e3 = np.exp(-x1 - 0.1)
    f = e1 + e2 + e3
    df_dx1 = e1 + e2 - e3
    df_dx2 = 3*(e1 - e2)
    g = np.array([df_dx1, df_dx2])
    if hessian:
        h11 = e1 + e2 + e3
        h12 = 3*(e1 - e2)
        h21 = h12
        h22 = 9*(e1 + e2)
        h = np.array([[h11, h12], [h21, h22]])
        return f, g, h
    else:
        return f, g, None
