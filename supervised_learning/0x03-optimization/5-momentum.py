#!/usr/bin/env python3
""" Function Updates Variables Momentum
alpha is the learning rate
beta1 is the momentum weight
var is a numpy.ndarray containing the variable to be updated
grad is a numpy.ndarray containing the gradient of var
v is the previous first moment of var
Returns: the updated variable and the new moment, respectively
"""


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Function Update Variables Momentum"""
    new_prom = (beta1 * v) + ((1 - beta1) * grad)
    new_var = var - alpha * new_prom
    return (new_var, new_prom)
