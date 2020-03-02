#!/usr/bin/env python3
"""Function Normalization constant
X is the numpy.ndarray of shape (d, nx) to normalize
d is the number of data points
nx is the number of features
m is a numpy.ndarray of shape (nx,) that contains the mean of all features of X
s is a numpy.ndarray of shape (nx,) that contains the standard deviation
of all features of X
Returns: The normalized X matrix
"""

import numpy as np


def normalize(X, m, s):
    """Function Normalization"""
    eps = 0.000000001
    norm = (X - m) / (np.sqrt(np.power(s, 2) + eps))
    return (norm)
