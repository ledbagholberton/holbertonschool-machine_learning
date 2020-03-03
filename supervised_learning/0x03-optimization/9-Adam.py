#!/usr/bin/env python3
"""  Function Adam Update Variables
updates a variable in place using the Adam optimization algorithm:

alpha is the learning rate
beta1 is the weight used for the first moment
beta2 is the weight used for the second moment
epsilon is a small number to avoid division by zero
var is a numpy.ndarray containing the variable to be updated
grad is a numpy.ndarray containing the gradient of var
v is the previous first moment of var
s is the previous second moment of var
t is the time step used for bias correction
Returns: the updated variable, the new first moment,
and the new second moment, respectively
"""


import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Function Update Variables Adam"""
    Vd = (beta1 * v) + ((1 - beta1) * grad)
    Sd = (beta2 * s) + ((1 - beta2) * grad * grad)

    Vd_corr = Vd / (1 - beta1 ** t)
    Sd_corr = Sd / (1 - beta2 ** t)

    w = var - alpha * (Vd_corr / ((Sd_corr ** (0.5)) + epsilon))
    return (w, Vd, Sd)
