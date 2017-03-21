import functools 
import matplotlib.pyplot as plt
import numpy as np


"""Code adapted from Data Science From Scratch"""

def predict(beta0, beta1, x_i):
    """Returns simple linear regression prediction on x_i given coefficients beta0 and beta1"""
    return beta1 * x_i + beta0

def error(beta0, beta1, x_i, y_i):
    """Returns residual error of simple linear regression prediction of
    y_i given x_i and coefficients beta0 and beta1"""
    return y_i - predict(beta0, beta1, x_i)

def sum_of_squared_errors(betas, x, y):
    """Returns sum of squared errors for simple linear regression model y = beta0 + beta1 x"""
    beta0, beta1 = betas
    return sum(error(beta0, beta1, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))

def make_cost_function(x, y):
    """Returns a function that computes sum_of_squared_errors as a function of betas
    treating x and y as fixed."""
    return functools.partial(sum_of_squared_errors, x=x, y=y)

def gradient_on_data(betas, x, y):
    """Returns the gradient at this point betas=(beta0, beta1) given dataset x and y"""
    beta0, beta1 = betas
    partial_deriv0 = -sum(2*(y_i - predict(beta0, beta1, x_i))
        for x_i, y_i in zip(x, y))
    partial_deriv1 = -sum(2*(y_i - predict(beta0, beta1, x_i))*x_i
        for x_i, y_i in zip(x, y))
    return (partial_deriv0, partial_deriv1)

def make_gradient(x, y):
    """Returns a function that computes the gradient at this point beta=(beta0, beta1) 
    treating x and y as fixed"""
    return functools.partial(gradient_on_data, x=x, y=y)

def cost_for_point(x_i, y_i, betas):
    return sum_of_squared_errors(betas, [x_i], [y_i])

def gradient_for_point(x_i, y_i, betas):
    return gradient_on_data(betas, [x_i], [y_i])

def make_contour_plot(J, xlow=5, xhigh=10, ylow=0.0, yhigh=0.1):
    beta0 = np.linspace(xlow, xhigh)
    beta1 = np.linspace(ylow, yhigh)
    js = [J([b0, b1]) for b0 in beta0 for b1 in beta1]   # J(beta0, beta1) for every beta0,beta1 pair
    js = np.reshape(js, (len(beta0),len(beta1)))
    js = js.transpose()
    plt.contour(beta0, beta1, js, 50)
    plt.xlabel('beta0')
    plt.ylabel('beta1')
    plt.title('J(beta0, beta1)')