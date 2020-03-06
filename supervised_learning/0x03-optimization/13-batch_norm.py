#!/usr/bin/env python3
"""  Function Batch Normalization
that normalizes an unactivated output of a neural network
using batch normalization:

Z is a numpy.ndarray of shape (m, n) that should be normalized
m is the number of data points
n is the number of features in Z
gamma is a numpy.ndarray of shape (1, n) containing the scales
used for batch normalization
beta is a numpy.ndarray of shape (1, n) containing the offsets
used for batch normalization
epsilon is a small number used to avoid division by zero
Returns: the normalized Z matrix
"""


import tensorflow as tf
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Function Batch Normalization"""
    Z_norm = ((Z - np.mean(Z, axis=0))
              / ((np.var(Z, axis=0) + epsilon)**0.5))
    return(gamma * Z_norm + beta)
