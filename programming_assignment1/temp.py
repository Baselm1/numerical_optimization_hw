#####################################################
#########   Unconstrained Optimizer Class   #########
#####################################################


# Dependencies:

import numpy as np 
import utils

# Class Description:

'''
Important Note:

    Note that everything that this class does can be done just as easily with functions.
    And on top of that, having a class is kind of redundant for our application since we'll only ever use one "optimizer" instance or object.
    But, with a class, we can encapsulate things very well and cleanly, and we can store the results of each iteration in the object and use that later for plotting in a much cleaner fashion.
    While the scope of this project isn't huge, it's always a good practice to optimize the code we write. 

Intended Usage:



'''

class UnconstrainedOptimizer:

    def __init__(self):
        '''
        Class Constructor:
            - Inputs: None
            - Attributes: 
                - path: A list to store all the points x_i for every i'th iteration.
                - objective_values: A list to store the objective functions value at x_i which is f(x_i) for every i'th iteration.
        '''
        self.path = []
        self.objective_values = []

    def gradient_descent(self):

        # First, set the current_x to x_0 as we start minimizing:
        current_x = self.x0

        # Also we get the current objective and the gradient of the function:
        current_f, current_gradient = self.f(self.x0)

        # Now we setup the loop as implemented in class with slight modifications:

        iterator = 0
        success = False

        while not success and iterator<=self.max_iter:
            
            # First, get the "learning rate" using linear search with Wolfe's condition:
            alpha = self.line_search(current_gradient)

            # Then, update the value of x, and check for convergence:
            next_x = current_x - alpha*current_gradient
            
            # Now we get the next objective and gradient and check for convergence:
            next_f , next_gradient = self.f(next_x)
            success = self.check_convergence(current_x, next_x, current_f, next_f, next_gradient)

            # In any case, we update current values to the next values for the next loop, however if success is True note that the loop breaks.
            current_x, current_f, current_gradient = next_x, next_f, next_gradient
            self.path.append(current_x)
            self.objective_values.append(current_f)
            iterator+=1
        
        # There are many ways to return things from this function, I chose to only return the success as I have access to the other stuff we want/ need.
        return success

    def newton_method(self):
        '''
        For Newton's method, we'll always start with the Hessian matrix as B0. 
        However, during my research on the subject, I found that in some implementations we can start with B0 = I_n and then update it from there for every iteration.
        This will be implemented in case a function returns a None type matrix or the Hessian isn't defined for the function.
        The base assumption of this implementation is that the functions defined in examples.py always return f,g and h as described in the HW document...
        And hence I'll always assume in this code that B0 = Hessian(f) and then we update accordingly. 

        NOTE: I'll ommit repeating comments from gradient descent and only comment what's new in this method.
        '''
        pass

 
        

    def line_search(self, x, gradient):
        
        # First we define constants as given. I'm assuming we're not using the curvature condition since we're only given one constant.

        RHO = 0.5 
        C1 = 0.01 

        # Now we start with an initial alpha and work from there:

        alpha = 1.0

        while True:
            next_x = None  
            pass
        return alpha

    def check_convergence(self, prev_x, next_x, prev_f, next_f, next_gradient):
        '''
        Description: Checks convergence based on the obj_tol and param_tol supplied by the user to the class.

        Inputs: 
            - prev_x, next_x: The 'params of the current and next iteration (or previous and current iteration).
            - prev_f, next_f: The 'obj' or objective of current and next iteration which is f(x_i)
        
        Output:
            - Returns True if both convergence conditions holds.
        '''
        pass

    def minimize(self, f, x0, method, obj_tol = 1e-12, param_tol=1e-8, max_iter=100):

        '''
        The minimize function is the "heart" of this class.

        Inputs:
            - f: The function to be optimized, for now assumed to return f,g and maybe h, where f is the function evaluated at x, g is the gradient evaluated at x, and h is the Hessian matrix.
            - x0: The starting point of the optimization.
            - method: A string for either method, up to the users choice.
            - obj_tol: The numeric tolerance for successful termination between two f(x_i) and f(x_{i+1}) steps, 1e-12 by default.
            - param_tol: The numeric tolerance for successful termination between two x_i and x_{i+1}, 1e-8 by default.
            - max_iter: The number of iterations, 100 by default.

        Outputs:
            - final_location: The final x_i.
            - final_objective: The final f(x_i)
            - succeeded: A boolean value indicating success or failure when minimizing a specific function. 
        '''

        # First: define the attributes the optimizing algorithms will use:

        self.f = f
        self.x0 = x0
        self.path.clear() # If it was full from previous run. 
        self.path = [x0] # Add the first point to the sequence of points.
        self.objective_values.clear() # If it was full from previous run.
        self.objective_values = [f(x0)] # Add the first objective value to the objective values for plotting.
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter

        # Now we check which method was chosen by the user. In case of an invalid case, we raise an error.

        if method == 'gradient_descent':
            success = self.gradient_descent()
        elif method == 'newton_method':
            success = self.newton_method()
        else:
            raise ValueError("The provided method is unsupported.")
        
        # Assuming the methods executed successfully:
        final_location, final_value = self.path[-1], self.objective_values[-1] 

        return final_location, final_value, success

    def plot_path(self):
        pass

