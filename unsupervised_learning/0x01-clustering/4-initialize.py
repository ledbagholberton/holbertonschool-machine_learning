#!/usr/bin/env python3
"""
Write a function  that i:

X is a numpy.ndarray of shape (n, d) containing the data set
k is a positive integer containing the number of clusters
You are not allowed to use any loops
Returns: pi, m, S, or None, None, None on failure
pi is a numpy.ndarray of shape (k,) containing the priors for
each cluster, initialized evenly
m is a numpy.ndarray of shape (k, d) containing the centroid means
for each cluster, initialized with K-means
S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
for each cluster, initialized as identity matrices
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model"""
    if not verify(X, k):
        return None, None, None
    n, d = X.shape
    m, _ = kmeans(X, k)
    pi = np.random.uniform(1/k, 1/k, size=(k, ))
    iD = np.identity(d)
    S = np.broadcast_to(iD, (k, d, d))
    # S = np.expand_dims(iD, axis=0)
    return(pi, m, S)


def verify(X, k):
    """verifiy conditions"""
    if not isinstance(X, np.ndarray):
        return False
    if type(k) is not int or k < 0:
        return False
    return True
