#!/usr/bin/env python3
"""
Write a function  that calculates the Q affinities:
Y is a numpy.ndarray of shape (n, ndim) containing the low dimensional
transformation of X
n is the number of points
ndim is the new dimensional representation of X
Returns: Q, num
Q is a numpy.ndarray of shape (n, n) containing the Q affinities
num is a numpy.ndarray of shape (n, n) containing the numerator
of the Q affinities
"""
import numpy as np


def Q_affinities(Y):
    """FUnction Q_Affinities"""
    n, ndim = Y.shape
    Q = np.zeros((n, n))
    sum_Y = np.sum(np.square(Y), 1, keepdims=True)
    D = sum_Y + sum_Y.T - 2*np.dot(Y, Y.T)
    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)
    return(Q, num)
