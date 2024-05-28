#####################################################
#########   Unconstrained Optimizer Class   #########
#####################################################


# Dependencies:

import numpy as np 

# Functions:

def gradient_descent(f, x0, alpha, max_iter, obj_tol, param_tol, line_search = False):

    # First: Setup
    current_x = x0
    current_f , current_gradient, _ = f(x0)
    points = [current_x]
    objectives = [current_f]
    iterator = 0
    success = False

    # Now we loop and minimize:
    while not success and iterator<=max_iter:

        # First we check if we need to update the step size.

        if line_search: # If we need to line search for a better alpha then we call line search here before checking the new point.
            alpha = line_search_bt(f, current_x, -current_gradient)
        
        # Now we move towards the gradient with the new step size, and set the new values to the current values for next loop.

        print(f'Current Iteration: {iterator}, Current x value: {current_x} and current objective value: {current_f}')
        next_x = current_x - alpha*current_gradient
        next_f, next_gradient, _ = f(next_x)
        iterator += 1 
        success = check_convergence(current_x, next_x, current_f, next_f, obj_tol, param_tol)
        current_x = next_x
        current_f = next_f
        current_gradient = next_gradient

        # Lastly, we append the new points reached to the lists so that we can return them for plotting.

        points.append(next_x)
        objectives.append(next_f)
    
    # Once we finish, we return everything that is needed.
    return points, objectives, success

'''
Note: For newtons method, we need to do many optimizations to insure that the algorithm is working optimally.
For this reason, at every step, we check if the Hessian matrix is definite positive.
If it is, we simply use it, if it's not, we use the BFGS approximation.
'''

def is_positive_definite(matrix):
    """
    Check if a matrix is positive definite.
    """
    return np.all(np.linalg.eigvals(matrix) > 0)

def BFGS(H, s, y):
    '''
    We update the B_{k+1} matrix as we saw in the lecture using the BFGS method if needed.
    '''
    first_frac = np.outer(np.dot(H, s), np.dot(s, H)) / np.dot(s, np.dot(H, s))
    second_frac = np.outer(y, y) / np.dot(y, s)
    return H - first_frac + second_frac

def newtons_method(f, x0, alpha, max_iter, obj_tol, param_tol, line_search=False):

    # As with gradient descent, first we setup the code beforte looping.
    current_x = x0
    current_f, current_gradient, current_hessian = f(x0, hessian=True)
    approx_hessian = False
    # Now we check if the Hessian matrix is not None or is definite positive.
    if current_hessian is None:
        approx_hessian = True # Allows our loop to approximate the hessian for every loop and ignore checking positive definite condition.
        current_hessian = np.eye(x0.shape[0]) # We start with the identity matrix.
    
    elif not is_positive_definite(current_hessian):
        current_hessian = np.eye(x0.shape[0]) # We start with the identity matrix.
    
    iterator = 0 
    success = False
    points = [current_x]
    objectives = [current_f]

    while not success and iterator <= max_iter:
        
        # First, we check if we need a line search alpha or to use a fixed alpha.
        p = -np.linalg.inv(current_hessian) @ current_gradient
        if line_search: 
            alpha = line_search_bt(f, current_x, p)
        
        # Now we update the new x and check for convergence.
        print(f'Current Iteration: {iterator}, Current x value: {current_x} and current objective value: {current_f}')
        new_x = current_x + alpha*p
        new_f, new_gradient, new_hessian = f(new_x, hessian=True)
        s = new_x - current_x
        y = new_gradient - current_gradient        
        if approx_hessian:
            new_hessian = BFGS(current_hessian, s, y)
        elif not is_positive_definite(new_hessian): # Redundant elif case, but saves on a function call to check if we don't need to check and we know we always get a None from the objective function.
            new_hessian = BFGS(current_hessian, s, y)

        success = check_convergence(current_x, new_x, current_f, new_f, obj_tol, param_tol)
        iterator+=1 
        current_x = new_x
        current_f = new_f
        
        # And lastly we append the new points to the arrays we want to return.

        points.append(new_x)
        objectives.append(new_f)
     
    return points, objectives, success

def line_search_bt(f, x, p, c1=0.01, backtrack_constant=0.5):
    '''
    Linesearch takes in 3 inputs: 
        - f: The objective function.
        - x: The x_k of the k'th iteration of either gradient descent or newtons method.
        - p: The p_k direction vector, which will be solved for in each method separately.
        And Optional inputs c1, the Wolfe constant, and a backtracking_constant (rho).

        It backtracks until it finds the best fitting alpha satisfying only the first Wolfe condition and returns that alpha.
    '''
    # First, check for valid inputs:

    if not c1>0 and c1<1:
        raise ValueError('The provided Wolfe constant is invalid.') 
    
    if not backtrack_constant>0 and backtrack_constant<1:
        raise ValueError('The provide backtracking constant in invalid')

    alpha = 1.0 # We start with an initial alpha. 
    fx , gradient, _ = f(x)
    while True:
        new_x = x + alpha * p  
        new_fx, _, _ = f(new_x)  
        # First Wolfe condition.
        if new_fx <= fx + (c1 * alpha * np.dot(gradient, p)):
            break
        # Reduce step size.
        alpha *= backtrack_constant
    return alpha 

def check_convergence(current_x, next_x, current_f, next_f, obj_tol, param_tol):
    obj_diff = np.abs(next_f - current_f)
    param_diff = np.linalg.norm(next_x - current_x)
    return (obj_diff<obj_tol) and (param_diff<param_tol)

def minimize(obj_function, x0, method, max_iter=100, ls=False, obj_tol=1e-12, param_tol=1e-8):
    
    points, objectives = [] , [] 
    success = None

    # First we check what method type was provided and then decide what function to call.

    if method == 'gradient_descent':
        points, objectives, success = gradient_descent(obj_function, x0, 0.1 ,max_iter, obj_tol, param_tol,line_search=ls)
    elif method == 'newtons_method':
        points, objectives, success = newtons_method(obj_function, x0, 0.01 ,max_iter, obj_tol, param_tol,line_search=ls)
    else:
        raise ValueError('Invalid method was provided')
    
    return points, objectives, success


