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


def pca(X, var=0.95):
    """Function PCA"""
    n, m = X.shape
    U, S, Vh = np.linalg.svd(X)
    sum_vars = np.sum(S)
    var_ret = S/sum_vars
    sum = 0
    count = 0
    for i in var_ret:
        sum += i
        count += 1
        if sum > var:
            break
    new_Vh = Vh.T[:, :count]
    return (new_Vh)
