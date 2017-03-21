from __future__ import division
from collections import Counter
from linear_algebra import distance, vector_subtract, scalar_multiply
import math, random


#
# Some helper functions
#

def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h

def partial_difference_quotient(f, v, i, h):

    # add h to just the i-th element of v
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]
         
    return (f(w) - f(v)) / h

def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h)
            for i, _ in enumerate(v)] 

def step(v, direction, step_size):
    """move step_size in the direction from v"""
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]

def safe(f):
    """define a new function that wraps f and return it"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print 'yikes', e
            return float('inf')         # this means "infinity" in Python
    return safe_f


#
# Gradient descent
#

def minimize_batch(target_fn, gradient_fn, theta_0, 
    tolerance=0.000001, 
    max_iterations=10000, 
    step_sizes=None, 
    stats=None):
    """
    This is a lightly modified version of the GD algorithm from the book.

    Uses gradient descent to find theta that minimizes target function
    """
    
    assert step_sizes is not None, "Step sizes must be initialized!"
    
    theta = theta_0                           # set theta to initial value
    target_fn = safe(target_fn)               # safe version of target_fn
    value = target_fn(theta)                  # value we're minimizing
    
    for iteration in range(max_iterations):

        gradient = gradient_fn(theta)  
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]


        # choose the one that minimizes the error function        
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        if iteration % 1000 == 0:
            print "Executing iteration", iteration
            # you may wish to add some debugging info here

        # stop if we're "converging"
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

    return theta

#
# Stochastic gradient descent
#

def in_random_order(data):
    """generator that returns the elements of data in random order"""
    indexes = [i for i, _ in enumerate(data)]  # create a list of indexes
    random.shuffle(indexes)                    # shuffle them
    for i in indexes:                          # return the data in that order
        yield data[i]

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, 
    alpha_0=0.01, 
    max_iterations=10000, 
    stats=None):
    """
    This is a lightly modified version of the SGD from the book.

    Uses stochastic gradient descent to find theta that minimizes target function
    """

    data = zip(x, y)
    theta = theta_0                             # initial guess
    alpha = alpha_0                             # initial step size
    min_theta, min_value = None, float("inf")   # the minimum so far
    iterations_with_no_improvement = 0

    for iteration in range(max_iterations):

        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data )

        if value < min_value:
            # if we've found a new minimum, remember it
            # and go back to the original step size
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # otherwise we're not improving, so try shrinking the step size
            iterations_with_no_improvement += 1
            alpha *= 0.9
            # if we ever go 100 iterations with no improvement, stop
            if iterations_with_no_improvement >= 100:
                print "Went 100 iterations with no improvement, stopping"
                break

        # and take a gradient step for each of the data points        
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))

    return min_theta

