#!/usr/bin/env python3
"""
Calculates the total intra-cluster variance for a data set:

X is a numpy.ndarray of shape (n, d) containing the data set
C is a numpy.ndarray of shape (k, d) containing the centroid
means for each cluster
Returns: var, or None on failure
var is the total variance
"""
import numpy as np


def variance(X, C):
    """Intracluster variance"""
    if not verify(X, C):
        return None
    distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
    closest = np.min(distances, axis=0)
    sum = np.sum(np.power(closest, 2))
    return sum


def verify(X, C):
    """Verify requirements"""
    if not isinstance(X, np.ndarray):
        return False
    if not isinstance(C, np.ndarray):
        return False
    if C.shape[1] is not X.shape[1]:
        return False
    return True
