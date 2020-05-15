#!/usr/bin/env python3
"""
Write a function  that performs PCA on a dataset:

X is a numpy.ndarray of shape (n, d) where:
n is the number of data points
d is the number of dimensions in each point
ndim is the new dimensionality of the transformed X
Returns: T, a numpy.ndarray of shape (n, ndim)
containing the transformed version of X
"""
import numpy as np


def pca(X, ndim):
    """Function PCA2 with dim"""
    m, n = X.shape
    X = np.mean(X, axis=0) - X
    U, Sigma, Vh = np.linalg.svd(X, full_matrices=True)
    new_Vh = Vh.T[:, :ndim]
    T = np.matmul(X, new_Vh)
    return (-1 * T)
