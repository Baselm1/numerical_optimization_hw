############################################3###
############ CONSTRAINED OPTIMIZER #############
################################################

'''
    ADD NOTES 
'''

import numpy as np

class InternalPointOptimzer:

    def __init__(self, objective_function: callable, ineq_constraints: list, A: np.ndarray, b: np.ndarray, x0: np.ndarray, grad_epsilon=1e-8, c1=1e-4, c2=0.9, rho=0.9, t=1, mu=10, max_iter=1000, ineq_tol=1e-9) -> None:
        '''
            Constructor method of the class.
            I provided additional params that can be changed for different optimization behavior.
            
            In case the names are not self expalantory - Params:
            - grad_epsilon: The epsilon value for which we calculate the gradient.
            - c1, c2 and rho: The constants for the backtracking line search.
            - t and mu: The starting constants for the log barrier.
            - ineq_tol: 
        '''
        self.f = objective_function
        self.ineq_constraints = ineq_constraints
        self.A = A
        self.b = b
        self.x0 = x0
        self.grad_epsilon = grad_epsilon
        self.c1 = c1
        self.c2 = c2
        self.rho = rho
        self.t = t
        self.mu = mu
        self.max_iter = max_iter
        self.ineq_tol = ineq_tol
        self.newton_tol = ineq_tol/10 
        self.path = []
        self.objectives = []

    def gradient(self, f: callable, x: np.ndarray, epsilon: float=1e-8):
        '''
            Gradient is only used if the provided function does not have an already analytically calculated Gradient.
            Note that this is a very simple "limit" evaluation close to the definition in calculus.
            To find the derivative in a computational manner we simply evaluate the limit at a value close to zero.

            Params:
                f: The function to find the gradient for.
                x: The point of where we evaluate the gradient.
                epsilon: Used in place of "h" in the limit definition, as we can't divide by 0.

            Returns:
                The approximate gradient of the provided function 'f' around the point 'x'.
        '''
        
        # Implementing the derivative approximation as found in Nocedal, pg.(195), eq.(8.1).
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            grad[i] = (f(x_plus) - f(x)) / epsilon
        return grad

    def BFGS(self, B: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
            BFGS is only used if the provided function does not have an already analytically calculated Hessian Matrix.

            Params:
                B: The Hessian (approximation) at the current iteration.
                s: The step vector at the current iteration.
                y: The gradient difference vector at the current iteration.
            
            Returns:
                An updated Hessian approximation. 
        '''

        # Implementing the update function using a compact representation, found in Nocedal, pg.(139) eq.(6.13). 
        rho = 1.0 / np.dot(y, s)
        I = np.eye(len(s))
        V = I - rho * np.outer(s, y)
        B_new = np.dot(V.T, np.dot(B, V)) + rho * np.outer(s, s)
        return B_new
    
    def get_gradient(self, f: callable, x: np.ndarray) -> np.ndarray:
        '''
            A wrapper function to keep the code clean inside of Newton's method.
            Returns the gradient of a function based on user input.
        '''
        gradient = None
        if self.compute:
            gradient = self.gradient(f, x)
            return gradient
        _, gradient = f(x)
        return gradient

    def get_hessian(self, f: callable, x: np.ndarray, *args) -> np.ndarray:
        '''
            A wrapper function to keep the code clean inside of Newton's method.
            Returns the Hessian matrix or approximation depending on user input.
            If compute = True, the arguments in *args is expected to be in the following order: (H, s, y).
        '''
        if self.compute:
            return self.BFGS(args[0], args[1], args[2])
        _, _, hessian = f(x, hessian=True)
        return hessian
    
    def f_call(self, f, x, hessian=False) -> tuple:
        '''
            Since function calls depend on the value of self.compute, this method acts as a wrapper to simplify function calls.
            Assumes this function won't be called at the start of newton method.

            Params:
                f: The function we want to call and get the values from.
                x: The point we want to evaluate the function at.
                Hessian: Returns the Hessian if needed.
            
            Returns:
                A tuple with (objective_value, gradient, hessian (or None)).
        '''

        if self.compute:
            objective = f(x)
            gradient = self.get_gradient(f,x)
            hessian_mat = self.get_hessian if hessian else None
            return (objective, gradient, hessian_mat)
        else:
            return f(x, hessian=hessian) # The "usual" return.

    def newton_method(self, f: callable) -> np.ndarray:
        '''
            Optimizes an unconstrained provided log-barrier function.

            Params:
                f: The log barrier modified function.
            
            Returns:
                The path of points the algorithm took to optimize.
        '''

        # Setup:

        # First we add the starting point to new arrays, since each call of newton method will have a different path it takes.
        path = np.zeros([self.max_iter + 1, len(self.x0)])
        path[0, :] = self.x0

        # Now we setup the start of the algorithm.
        self.converged = False
        iteration = 0
        current_x = self.x0
        if self.compute:
            current_f = f(current_x)
            current_gradient = self.get_gradient(f, current_x)
            current_hessian = np.eye(self.x0.shape[0])
        else:
            current_f, current_gradient, current_hessian = f(current_x, hessian=True)
        eq_constraints = np.shape(self.A)[0]

        # We loop until convergence or max iterations:
        while iteration < self.max_iter and (not self.converged):
            
            # Evaluate the function, gradient, and hessian for the current loop.
            if not iteration==0:
                current_f, current_gradient, current_hessian = self.f_call(f, current_x, hessian=True)
                
            
            # Solve the KKT system:
            solution = self.solveKKT(eq_constraints, current_hessian, current_gradient)

            # Get the direction vector:
            pk = solution[0:len(self.x0)]

            # Perform line search:
            alpha = self.backtracking_line_search(f, pk, current_x)
            new_x = current_x + alpha * pk

            # Add the new found x to the path and check for convergence.
            path[iteration + 1, :] = new_x
            new_f = self.f_call(f, new_x, hessian=False)[0]

            # Check convergence and return results if converged.
            newton_decrement = np.sqrt(np.dot(pk, np.dot(current_hessian, pk)))
            if (newton_decrement**2)/2 < self.newton_tol:
                self.converged = True
                path = np.delete(path, (np.arange(iteration + 1, len(path))), 0)
                print(f"Convergence achieved at iteration: {iteration}, Function value: {new_f}, At: {new_x}")
                return path
            
            # Otherwise we set up the variables for the next loop.
            current_x = new_x
            iteration += 1

        return path

    def solveKKT(self, eq_constraints: int, H: np.ndarray, grad: np.ndarray) -> np.ndarray:
        '''
            Constructs and solves the KKT system.
            Note that if the inequality constraints don't exist then the system is much simpler to solve.

            Params:
                eq_constraints: The number of variables of equality constraints.
                H: The current Hessian matrix of the current loop.
                grad: The current Gradient of the current loop.

            Returns:
                The solution to the system Ax=b that represents the KKT system. 
        '''
        
        if eq_constraints > 0: # Check if we have equality constraints:
            # The KKT Matrix:
            A_eq = self.A
            A_t = np.transpose(A_eq)
            zero_block = np.zeros((eq_constraints, eq_constraints))
            kkt_mat = np.array([
                [H, A_t],
                [A_eq, zero_block]
            ])
            # The RHS vector:
            kkt_rhs = np.concatenate((-grad, np.zeros(eq_constraints)))
        else:
            # If there are no equality constraints, use only the Hessian and gradient
            kkt_mat = H
            kkt_rhs = -grad
            
        return np.linalg.solve(kkt_mat, kkt_rhs) # Solve and return the solution.

    def backtracking_line_search(self, f: callable, pk: np.ndarray, x: np.ndarray) -> float:
        '''
            Calculates the best step size and returns it.
        '''
        alpha = 1.0
        objective, gradient, _ = self.f_call(f, x)
        while True:
            new_x = x + alpha * pk
            new_f, new_grad, _ = self.f_call(f, new_x)

            # Wolfe conditions:
            if new_f <= objective + self.c1 * alpha * np.dot(gradient, pk) \
                and (not self.compute or np.dot(new_grad, pk) >= self.c2 * np.dot(gradient, pk)):
                return alpha

            alpha *= self.rho

    def log_barrier(self) -> callable:
        '''
            Defines and returns a function that when called returns the proper values as needed.
        '''
        t = self.t

        # If self.compute is true, then we know the function doesn't return neither the gradient nor the hessian:
        if self.compute:
            def phi(x: np.ndarray, *args) -> float:
                func_term = t*self.f(x)
                
                log_term = -1*sum(np.log(-1*g(x)) for g in self.ineq_constraints)
                return func_term + log_term
        else:
            def phi(x: np.ndarray, hessian: bool=False) -> tuple:
                outputs = self.f(x, hessian)
                objective = outputs[0] * t
                gradient = outputs[1] * t
                if hessian:
                    hessian_mat = outputs[2] * t
                    for g in self.ineq_constraints:
                        objective_i, gradient_i = g(x)
                        objective -= np.log(-objective_i)
                        gradient -= gradient_i / objective_i
                        hessian_mat += np.outer(gradient_i, gradient_i) / (objective_i ** 2)

                    return objective, gradient, hessian_mat
                else:
                    for g in self.ineq_constraints:
                        objective_i, gradient_i = g(x)
                        objective -= np.log(-objective_i)
                        gradient -= gradient_i / objective_i
                    return objective, gradient
                
        return phi
    
    def solve(self, compute=False):
        self.compute = compute
        log_barrier_func = self.log_barrier()
        
        while True:
            optimization_path = self.newton_method(log_barrier_func)
            
            # Check convergence based on objective function value change
            current_objective = self.f_call(log_barrier_func, optimization_path[-1])[0]
            previous_objective = self.f_call(log_barrier_func, optimization_path[-2])[0] if len(optimization_path) > 1 else float('inf')
            
            if abs(current_objective - previous_objective) < self.ineq_tol:
                print("Converged.")
                break
            
            self.t *= self.mu
            self.x0 = optimization_path[-1, :]
            self.path = np.append(self.path, optimization_path[1:, :], axis=0)
    # def solve(self, compute=False) -> None:
    #     '''
    #         Main driver for the optimization class, this function mainly does the "outloops" of the algorithm and only finds log-barrier functions to optimize using newtons method.

    #         Params:
    #             compute: Set the computation type. False -> the gradient and hessian are already analytically computed and provided and don't need to be computed.
    #     '''

    #     # Set the computation type of the function and get the function we're starting with.
    #     self.compute = compute
    #     log_barrier_func = self.log_barrier()

    #     # Now we loop through the optimization, and in each loop if we don't converge we increase the value of t by a factor of mu.
    #     while len(self.ineq_constraints)/self.t >= self.ineq_tol:
            
    #         # 1) Optimize current barrier function: 
    #         optimization_path = self.newton_method(log_barrier_func)
            
    #         # 2) Check if we converged:
    #         if not self.converged:
    #             self.path = np.append(self.path, optimization_path[1:, :], axis=0)
    #             return # Exit the optimization
            
    #         # 3) Otherwise we continue optimizing as usual:
    #         self.t *= self.mu
    #         log_barrier_func = self.log_barrier()
    #         self.x0 = optimization_path[-1, :]
    #         self.path = np.append(self.path, optimization_path[1:, :], axis=0)
        
    #     # ADD PRINT STATEMENTS HERE AS REQUIRED FOR THE REPORT.
