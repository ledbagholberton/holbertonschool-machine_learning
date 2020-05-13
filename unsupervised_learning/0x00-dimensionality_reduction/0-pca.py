#!/usr/bin/env python3
"""
Write a function that performs PCA on a dataset
X is a numpy.ndarray of shape (n, d) where:
n is the number of data points
d is the number of dimensions in each point
all dimensions have a mean of 0 across all data points
var is the fraction of the variance that the PCA transformation
should maintain:
Returns: the weights matrix, W, that maintains var fraction of
X‘s original variance
W is a numpy.ndarray of shape (d, nd) where nd is the new
dimensionality of the transformed X
"""
import numpy as np


def pca(X, var=0.953):
    """Function PCA"""
    n, m = X.shape
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n-1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    # Sort eigen vals & eigen vectors in ascendent order
    key = np.argsort(eigen_vals)[::-1]
    eigen_vals, eigen_vecs = eigen_vals[key], eigen_vecs[:, key]
    # Sum of variances
    sum_vals = np.sum(eigen_vals)
    var_ret = eigen_vals/sum_vals
    sum = 0
    count = 0
    for i in var_ret:
        sum += i
        count += 1
        if sum > var:
            break
    new_eigen_vecs = eigen_vecs[:, :count]


    # Project X onto PC space
    return -1 * new_eigen_vecs
