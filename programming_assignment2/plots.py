import tests.examples as examples 
import src.constrained_min as m 
import numpy as np 


# Example 1 Plots: Analytical QP 

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
optimizer.plot()
optimizer.plot_line()


#Example 2 Plots: Analytical LP

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
optimizer.plot()
optimizer.plot_line()


# # Example 3 Plots: Numerical QP

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
optimizer.plot()
optimizer.plot_line()



# Example 3 Plots: Numerical LP

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
optimizer.plot()
optimizer.plot_line()


