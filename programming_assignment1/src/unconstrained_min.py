#####################################################
#########   Unconstrained Optimizer Class   #########
#####################################################


# Dependencies:

import numpy as np 
# import utils

# Class:

'''
    Notes:
        Initial implementation started with a class.
        However, as I didn't have a clear picture of what I exactly wanted, I implemented everything using functions.
        Python, being naturally without function overriding and pointers, resulted in a mess.
        Too many things are passed around too many times and functions names clashed with optional function parameters.
        This is why I'm re-writing the code in one encapsulated and clean class that covers everything with proper documentations and naming convensions, new variable names that are better and more.
        You can examine the previous mess in the previous commit on Github.
    
    Other Assumptions:
        When running Quasi-Newton methods, at every loop I checked if we needed a positive definite Hessian.
        In case the function does not return a Hessian matrix at all, like during the "setuo" before looping, I opted to "memorize" that with a boolean so that I don't check if the matrix is positive definite and ALWAYS update the Hessian approximation using BFGS.
        If however we do have a Hessian matrix during the "setup", we check if it's positive definite, if not we then again always update using BFGS.
        Lastly, if we start with a Hessian matrix but at some point during the run it's not positive definite, we replace the k'th step Hessian matrix with the BFGS approximation, and otherwise we use the Hessian of the function.
'''

class UnconstrainedOptimizer:

    ##########################
    ###### CONSTRUCTOR #######
    ##########################

    def __init__(self, alpha=0.01, obj_tol=1e-12, param_tol=1e-8, wolfe_constant=0.01, rho=0.5 ):
        
        '''
            The constructor builds a minimizer/ optimizer object.
            It takes in many of the different parameters that will not be changed between runs.
            
            Inputs:
                - alpha: Fixed learning rate for both minimizing algorithms when run without line search.
                - obj_tol: The difference tolerance between every pair of objectives f(x_i) and f(x_{i+1}) for every iteration in both algorithms.
                - param_tol: The difference tolerance between every pair of inputs x_i and x_{i+1} for every iteration in both algorithms.
                - Wolfe_constant: The constant for Wolfe's first condition, c1, which will be used in line search.
                - Rho: The backtracking constant which will be used in line search.
            
            Other Attributes:
                - path: A list of all points x_i for every 0<=i<=max_iter for both algorithms.
                - objectives: A list of all objectives f(x_i) for every 0<=i<=max_iter for both algorithms.
                - success: A boolean signifying the convergence of the function or not.
            
            Notes: 
                - All of the input parameters are optional, but the default values are as given in the document.
                - The choice of alpha=0.01 is arbitrary, for some functions it's more than enough to converge while for others 0.01 is a large step. This will only be used to demonstrate the different runs with and without line search enabled.
        '''

        # Check if we're given proper parameters:

        if not (wolfe_constant>0 and wolfe_constant<1):
            raise ValueError('Invalid Wolfe\'s first condition constant! Values must be in 0<c1<1.')
        if not (alpha>0 and alpha <1):
            raise ValueError('Invalid fixed step size! Values must be in 0<alpha<1.')
        if not (rho>0 and rho<1):
            raise ValueError('Invalid backtracking constant! Values must be in 0<rho<1.')
        
        # Otherwise, define all attributes:

        self.alpha = alpha
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.wolfe_constant = wolfe_constant
        self.rho = rho
        self.path = list()
        self.objectives = list()
        self.success = False

    ###############################
    ###### GRADIENT DESCENT #######
    ###############################

    def gradient_descent(self):

        # First: Setup before looping.
        current_x = self.x0
        current_f, current_gradient, _ = self.obj_f(self.x0)
        self.path.append(current_x)
        self.objectives.append(current_f)
        iterator = 0
        self.success = False # In case of second run call.

        # Now we loop untl convergence or failure.
        while not self.success and iterator<=self.max_iter:
            p = -current_gradient # Direction vector.
            if self.ls:
                self.alpha = self.line_search(current_x, p)
            if self.printing:
                print(f'Current Iteration: {iterator}, Current x: {current_x} and Current objective: {current_f}')
            next_x = current_x + self.alpha*p
            next_f, next_gradient, _ = self.obj_f(next_x)
            self.check_convergence(current_x, next_x, current_f, next_f)
            current_x = next_x
            current_f = next_f
            current_gradient = next_gradient
            self.path.append(next_x)
            self.objectives.append(next_f)
            iterator+=1

    ##############################
    ###### NEWTON'S METHOD #######
    ##############################

    def newton_method(self):

        # First: Setup before looping.
        current_x = self.x0
        current_f, current_gradient, current_hessian = self.obj_f(current_x)
        approx_hessian = False # Boolean set to always approximate the Hessian in case we need to.

        # We check if the defined function returns a proper Hessian matrix.
        if current_hessian is None:
            approx_hessian = True
            current_hessian = np.eye(current_x.shape[0]) # We start with the identity matrix.
        if not self.is_positive_definite(current_hessian):
            current_hessian = np.eye(current_x.shape[0])
        
        iterator = 0
        self.success = False
        self.path.append(current_x)
        self.objectives.append(current_f)

        while not self.success and iterator<=self.max_iter:
            p = -np.linalg.inv(current_hessian)@current_gradient # Inverse exist as we confirmed that the Hessian is not signular.
            if self.ls:
                self.alpha = self.line_search(current_x, p)
            if self.printing:
                print(f'Current Iteration: {iterator}, Current x value: {current_x} and current objective value: {current_f}')
            next_x = current_x + self.alpha*p
            next_f, next_gradient, next_hessian = self.obj_f(next_x, hessian=True)
            s = next_x - current_x
            y = next_gradient - current_gradient
            if approx_hessian:
                next_hessian = self.BFGS(current_hessian, s, y)
            elif not self.is_positive_definite(next_hessian):
                next_hessian = self.BFGS(current_hessian, s, y)
            self.check_convergence(current_x, next_x, current_f, next_f)
            current_x = next_x
            current_f = next_f
            current_gradient = next_gradient
            self.path.append(next_x)
            self.objectives.append(next_f)
            iterator+=1 

    ##############################
    ###### OTHER FUNCTIONS #######
    ##############################
    
    def is_positive_definite(self, matrix):
        
        '''
            Given a matrix, check if it's a positive definite matrix.
        '''

        return np.all(np.linalg.eigvals(matrix) > 0)
    
    def BFGS(self, H, s, y):

        '''
            Given the inputs:
                - H: The Hessian matrix or its approximate Bk at the k'th iteration of newtons method.
                - s: The difference between current and next x values.
                - y: The difference between current and next gradient.
            
            We run the update formula for BGS to obtain the next Hessian approximation.
        '''
        
        first_frac = np.outer(np.dot(H, s), np.dot(s, H)) / np.dot(s, np.dot(H, s))
        second_frac = np.outer(y, y) / np.dot(y, s)
        return H - first_frac + second_frac


    def line_search(self, x, p):

        '''
            Given the inputs:
            - x: The x_k of the k'th iteration of either algorithm.
            - p: The p_k direction of the k'th iteration of either algorithm.
            
            We line search for the best alpha values based on Wolfe's condition.

            Returns:
            - alpha: The best alpha value for the k'th iteration
        '''
        alpha = 1.0 # A good initial start. We can also use self.alpha which is the class's fixed step size.
        current_f, current_gradient, _ = self.obj_f(x)
        while True:
            next_x = x + alpha*p 
            next_f, _, _ = self.obj_f(next_x)
            if next_f <= current_f + (self.wolfe_constant * alpha * np.dot(current_gradient, p)): # First Wolfe condition.
                break
            alpha *= self.rho # Update alpha otherwise.
        return alpha

    def check_convergence(self, current_x, next_x, current_f, next_f):
        obj_diff = np.abs(next_f - current_f)
        param_diff = np.linalg.norm(next_x - current_x)
        if self.printing:
            print(f'Objective difference: {obj_diff}, Parameter difference: {param_diff}') 
        self.success = ((obj_diff<self.obj_tol) and (param_diff<self.param_tol))

    ##########################
    ###### MAIN DRIVER #######
    ##########################

    def minimize(self, objective_function, x0, method, line_search=True, printing=False, alpha=0.01,  max_iter=100):

        '''
            The minimize() function is the main driver for input functions.
            For every call of this function, we need to initialize a few things in case of previous runs.
            
            Inputs:
                - objective_function: The function to minimize. Assumed it returns f,g,h evaluated at input x (h can be None type).
                - x0: The starting point for the optimization.
                - method: Can either be 'gradient' or 'newton', and it's the selection of the user
                - line_search: A boolean selected by the user to either use line search or not (optional), True by default.
                - max_iter: The maximum number of allowe iterations (optional), 100 is the default value.
            
            Returns:
                - self.path
                - self.objectives
                - success: Boolean signifying if the function converged or not.
        '''

        self.obj_f = objective_function
        self.x0 = x0 
        self.ls = line_search
        self.max_iter = max_iter
        self.printing = printing # Set to true if you want a print for every iteration.

        # Initialize if minimize() was called multiple times on the same object:
        self.alpha = alpha 
        self.path.clear()
        self.objectives.clear()

        # Check what method was chosen:

        if method == 'gradient':
            self.gradient_descent()
        elif method == 'newton':
            self.newton_method()
        else:
            raise ValueError('Invalid method')
        
        # Print the last param and objective and the success:

        print(f'Terminated. Current point: {self.path[-1]}, Current objective: {self.objectives[-1]}, Succeeded: {self.success}')

        # return self.path, self.objectives, self.success

    