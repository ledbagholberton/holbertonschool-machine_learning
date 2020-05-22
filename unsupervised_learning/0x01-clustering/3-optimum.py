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
    varsi = []
    c, _ = kmeans(X, kmin, iterations)
    high_var = variance(X, c)
    for iter in range(kmin, kmax+1, 1):
        centroids, clss = kmeans(X, iter, iterations)
        results.append((centroids, clss))
        var = variance(X, centroids)
        varsi.append(var)
    d_vars = [varsi[0] - var for var in varsi]
    return(results, d_vars)


def verify(X, kmin, kmax, iterations):
    """verifiy conditions"""
    if not isinstance(X, np.ndarray):
        return False
    if len(X.shape) is not 2:
        return False
    if type(kmin) is not int or kmin <= 0:
        return False
    if X.shape[0] <= kmin:
        return False
    if type(kmax) is not int or kmax <= 0:
        return False
    if X.shape[0] <= kmax:
        return False
    if kmin >= kmax:
        return False
    if type(iterations) is not int or iterations <= 0:
        return False
    return True
