#!/usr/bin/env python3
"""
Write a function def cost(P, Q): that calculates the cost of
the t-SNE transformation:

P is a numpy.ndarray of shape (n, n) containing the P affinities
Q is a numpy.ndarray of shape (n, n) containing the Q affinities
Returns: C, the cost of the transformation
Hint: Watch out for division by 0 errors! Take the minimum of all
values, and almost 0 (ex. 1e-12)
"""
import numpy as np


def cost(P, Q):
    """Function cost"""
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    C = np.sum(P * np.log(P / Q))
    return C
