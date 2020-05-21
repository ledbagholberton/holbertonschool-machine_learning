#!/usr/bin/env python3
"""
tests for the optimum number of clusters by variance:

X is a numpy.ndarray of shape (n, d) containing the data set
kmin is a positive integer containing
the minimum number of clusters to check for (inclusive)
kmax is a positive integer containing
the maximum number of clusters to check for (inclusive)
iterations is a positive integer containing
the maximum number of iterations for K-means
Returns: results, d_vars, or None, None on failure
results is a list containing the outputs of K-means for each cluster size
d_vars is a list containing the difference in variance
from the smallest cluster size for each cluster size
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Optimize k-meas by variance intracluster"""
    if not verify(X, kmin, kmax, iterations):
        return None, None, None
    results = []
    d_vars = []
    c, _ = kmeans(X, kmin, iterations)
    high_var = variance(X, c)
    for iter in range(kmin, kmax+1, 1):
        results.append(kmeans(X, iter, iterations))
        centroids, _ = kmeans(X, iter, iterations)
        var = variance(X, centroids)
        d_vars.append(high_var - var)
    return(results, d_vars)


def verify(X, kmin, kmax, iterations):
    """verifiy conditions"""
    if not isinstance(X, np.ndarray):
        return False
    if type(kmin) is not int or kmin < 0:
        return False
    if type(kmax) is not int or kmax < 0:
        return False
    if kmax < kmin:
        return False
    if type(iterations) is not int or iterations <= 1:
        return False
    return True
