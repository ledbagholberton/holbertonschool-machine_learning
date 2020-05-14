#!/usr/bin/env python3
"""
Write a function  that calculates the Q affinities:

Y is a numpy.ndarray of shape (n, ndim) containing the low dimensional
transformation of X
n is the number of points
ndim is the new dimensional representation of X
Returns: Q, num
Q is a numpy.ndarray of shape (n, n) containing the Q affinities
num is a numpy.ndarray of shape (n, n) containing the numerator of the Q affinities
"""
import numpy as np

def Q_affinities(Y):
    """FUnction Q_Affinities"""
    n, ndim = Y.shape
    Q = np.zeros((n, n))
    sum_Y = np.sum(np.square(Y), 1, keepdims=True)
    suqared_d = sum_Y + sum_Y.T - 2*np.dot(Y, Y.T)
    D = suqared_d.clip(min=0)
    num = np.exp(-1 * D)
    for i in range(n):
        Q_iter = np.delete(num[i], i, axis=0)
        Q_iter = Q_iter / np.sum(Q_iter)
        Q_iter = np.insert(Q_iter, i, 0, axis=0)
        print(Q_iter.shape)
        Q[i] = Q_iter
    Q = (Q + Q.T) / (2*n)
    return(Q, num)
