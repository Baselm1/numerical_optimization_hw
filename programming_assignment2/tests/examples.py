############################################
############ EXAMPLE FUNCTIONS #############
############################################

'''
    In this file we only implement the example functions as describes, passing their constraints as a list of functions. 
    The file includes both examples for analytical and numerical methods.
    The class for the inequality constraints were inspired by: https://github.com/shira-shafir/constrained_opt/commits?author=shira-shafir
'''

import numpy as np

# Constraint Class:

class InequalityConstraint:
    def __init__(self, a, b, compute):
        self.a = a
        self.b = b
        self.compute = compute

    def __call__(self, x):
        val = np.dot(self.a, x) - self.b
        if self.compute:
            return val
        else:
            grad = self.a
            return val, grad

'''
    Analytical functions
'''

def constrained_problem_qp():

    # Define the function and return it as a callable:
    def func_qp(x, hessian=False):
        val = np.array(x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2)
        gradient = np.array([2 * x[0], 2 * x[1], 2 * (x[2] + 1)])
        if hessian:
            hessian_mat = 2 * np.eye(3)
            return val, gradient, hessian_mat
        else:
            return val, gradient
        
    #Define the constraints: 
    constraint1 = InequalityConstraint([-1, 0, 0], 0, compute=False)
    constraint2 = InequalityConstraint([0, -1, 0], 0, compute=False)
    constraint3 = InequalityConstraint([0, 0, -1], 0, compute=False)
    ineq_constraints = [constraint1, constraint2, constraint3]
    eq_constraints_mat = np.array([[1, 1, 1]]) 
    eq_constraints_rhs = np.array([1])
    
    # Define the starting point provided in the HW document.
    x0 = np.array([0.1, 0.2, 0.7])
    
    #Return the setup.
    return func_qp, x0, ineq_constraints, eq_constraints_mat, eq_constraints_rhs

def constrained_problem_lp():

    # Define the function and return it as a callable:
    def func_lp(x, hessian=False):
        val = - np.array(x[0] + x[1])  # max(x + y) is -min(x+y)
        gradient = - np.array([1.0, 1.0])
        if hessian:
            hessian_mat = np.zeros((2, 2))
            return val, gradient, hessian_mat
        else:
            return val, gradient
    
    #Define the constraints: 
    constraint1 = InequalityConstraint([-1, -1], -1, compute=False)
    constraint2 = InequalityConstraint([0, 1], 1, compute=False)
    constraint3 = InequalityConstraint([1, 0], 2, compute=False)
    constraint4 = InequalityConstraint([0, -1], 0, compute=False) 
    ineq_constraints = [constraint1, constraint2, constraint3, constraint4]
    eq_constraints_mat = np.array([])
    eq_constraints_rhs = np.array([])

    # Define the starting point provided in the HW document.
    x0 = np.array([0.5, 0.75])

    #Return the setup.
    return func_lp, x0, ineq_constraints, eq_constraints_mat, eq_constraints_rhs

'''
    Numerical functions (Less accurate results + Computationally expensive)
'''

def constrained_problem_qp_ver2():

    # Define the function and return it as a callable:
    def func_qp(x, hessian=False):
        return np.array(x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2)
    
    #Define the constraints: 
    constraint1 = InequalityConstraint([-1, 0, 0], 0, compute=True) 
    constraint2 = InequalityConstraint([0, -1, 0], 0, compute=True) 
    constraint3 = InequalityConstraint([0, 0, -1], 0, compute=True) 
    ineq_constraints = [constraint1, constraint2, constraint3]
    eq_constraints_mat = np.array([[1, 1, 1]]) 
    eq_constraints_rhs = np.array([1])

    # Define the starting point provided in the HW document.
    x0 = np.array([0.1, 0.2, 0.7])

    #Return the setup.
    return func_qp, x0, ineq_constraints, eq_constraints_mat, eq_constraints_rhs

def constrained_problem_lp_ver2():

    # Define the function and return it as a callable:
    def func_lp(x, hessian=False):
        return - np.array(x[0] + x[1])  # max(x + y) is -min(x+y)
    
    #Define the constraints: 
    constraint1 = InequalityConstraint([-1, -1], -1, compute=True)
    constraint2 = InequalityConstraint([0, 1], 1, compute=True)
    constraint3 = InequalityConstraint([1, 0], 2, compute=True)
    constraint4 = InequalityConstraint([0, -1], 0, compute=True) 
    ineq_constraints = [constraint1, constraint2, constraint3, constraint4]
    eq_constraints_mat = np.array([])
    eq_constraints_rhs = np.array([])

    # Define the starting point provided in the HW document.
    x0 = np.array([0.5, 0.75])

    #Return the setup.
    return func_lp, x0, ineq_constraints, eq_constraints_mat, eq_constraints_rhs