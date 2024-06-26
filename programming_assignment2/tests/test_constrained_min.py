import unittest
import numpy as np
import examples

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import constrained_min as m


class TestConstrainedMin(unittest.TestCase):

    # Recall the function when called returns: 
    # (f: Callable, x0: np.ndarray, ineq_constraints: list, eq_constraint_mat: np.ndarray, eq_constraint_rhs: np.ndarray)

    def test_qp_analytical(self):
        problem_params = examples.constrained_problem_qp()
        optimizer = m.InternalPointOptimzer(
            objective_function = problem_params[0],
            x0 = problem_params[1],
            ineq_constraints = problem_params[2],
            A = problem_params[3],
            b = problem_params[4],
            # Adjust the following params as needed.
            c1 = 1e-4, # Wolfe's sufficient condition.
            c2 = 0.1, # Wolfe's curvature condition.
            rho = 0.71, # Backtracking constant.
            t = 1, # Log barrier multiplier.
            mu = 10, # Factor of log barrier multiplier.
            max_iter = 300 # Newtons method max iteration.
        )
        optimizer.solve(compute=False)

        return 0

    def test_lp_analytical(self):
        problem_params = examples.constrained_problem_lp()
        optimizer = m.InternalPointOptimzer(
            objective_function = problem_params[0],
            x0 = problem_params[1],
            ineq_constraints = problem_params[2],
            A = problem_params[3],
            b = problem_params[4],
            # Adjust the following params as needed.
            c1 = 1e-4, # Wolfe's sufficient condition.
            c2 = 0.1, # Wolfe's curvature condition.
            rho = 0.71, # Backtracking constant.
            t = 1, # Log barrier multiplier.
            mu = 10, # Factor of log barrier multiplier.
            max_iter = 300 # Newtons method max iteration.
        )
        optimizer.solve(compute=False)

        return 0
    
    def test_qp(self):
        problem_params = examples.constrained_problem_qp_ver2()
        optimizer = m.InternalPointOptimzer(
            objective_function = problem_params[0],
            x0 = problem_params[1],
            ineq_constraints = problem_params[2],
            A = problem_params[3],
            b = problem_params[4],
            # Adjust the following params as needed.
            c1 = 1e-6, # Wolfe's sufficient condition.
            c2 = 0.5, # Wolfe's curvature condition.
            rho = 0.74, # Backtracking constant.
            t = 1, # Log barrier multiplier.
            mu = 10, # Factor of log barrier multiplier.
            max_iter = 500 # Newtons method max iteration.
        )
        optimizer.solve(compute=True)

        return 0

    def test_lp(self):
        problem_params = examples.constrained_problem_lp_ver2()
        optimizer = m.InternalPointOptimzer(
            objective_function = problem_params[0],
            x0 = problem_params[1],
            ineq_constraints = problem_params[2],
            A = problem_params[3],
            b = problem_params[4],
            # Adjust the following params as needed.
            c1 = 1e-4, # Wolfe's sufficient condition.
            c2 = 0.1, # Wolfe's curvature condition.
            rho = 0.45, # Backtracking constant.
            t = 1, # Log barrier multiplier.
            mu = 10, # Factor of log barrier multiplier.
            max_iter = 1000 # Newtons method max iteration.
        )
        optimizer.solve(compute=True)

        return 0


if __name__ == '__main__':
    unittest.main()