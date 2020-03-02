#!/usr/bin/env python3
""" Function shuffle data
X is the first numpy.ndarray of shape (m, nx) to shuffle
m is the number of data points
nx is the number of features in X
Y is the second numpy.ndarray of shape (m, ny) to shuffle
m is the same number of data points as in X
ny is the number of features in Y
Returns: the shuffled X and Y matrices
"""

import numpy as np


def shuffle_data(X, Y):
    """Function shuffle"""
    xper = np.random.permutation(X)
    yper = np.random.permutation(Y)
    return (xper, yper)
