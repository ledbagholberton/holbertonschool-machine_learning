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
    β1 = beta1
    β2 = beta2
    α = alpha
    ε = epsilon

    Vd = (β1 * v) + ((1 - β1) * grad)
    Sd = (β2 * s) + ((1 - β2) * grad * grad)

    Vd_ok = Vd / (1 - β1 ** t)
    Sd_ok = Sd / (1 - β2 ** t)

    w = var - α * (Vd_ok / ((Sd_ok ** (0.5)) + ε))
    return (w, Vd, Sd)

"""
    new_prom = (beta1 * v) + ((1 - beta1) * grad)
    new_s = (beta2 * s) + ((1 - beta2) * grad * grad)
    bias_correction1 = 1 - beta1 ** t
    new_prom_corr = new_prom / bias_correction1
    bias_correction2 = 1 - beta2 ** t
    new_s_corr = new_s / bias_correction2
    new_var = var - alpha * (new_prom_corr / ((new_s_corr ** (0.5)) + epsilon))
    return (new_var, new_s_corr, new_prom_corr)
"""