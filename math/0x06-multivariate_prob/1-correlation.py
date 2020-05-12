#!/usr/bin/env python3
"""
C is a numpy.ndarray of shape (d, d) containing a covariance matrix
d is the number of dimensions
If C is not a numpy.ndarray, raise a TypeError with the message C must
be a numpy.ndarray
If C does not have shape (d, d), raise a ValueError with the message C
must be a 2D square matrix
Returns a numpy.ndarray of shape (d, d) containing the correlation matrix
"""
import numpy as np


def correlation(C):
    """Function correlation"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    elif not ((len(C.shape) is 2) and (C.shape[0] is C.shape[1])):
        raise ValueError("C must be a 2D square matrix")
    else:
        v = np.sqrt(np.diag(C))
        outer_v = np.outer(v, v)
        correlation = C / outer_v
        correlation[C == 0] = 0
        return correlation
