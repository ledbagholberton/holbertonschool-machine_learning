#!/usr/bin/env python3
"""
Write a function  that :

X is a numpy.ndarray of shape (n, d) containing the data set
pi is a numpy.ndarray of shape (k,) containing the priors
for each cluster
m is a numpy.ndarray of shape (k, d) containing the centroid means
for each cluster
S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
for each cluster
You may use at most 1 loop
Returns: g, l, or None, None on failure
g is a numpy.ndarray of shape (k, n) containing the posterior probabilities
for each data point in each cluster
l is the total log likelihood
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm for a GMM"""
    try:
        if not verify(X, pi, m, S):
            return None, None
        n, d = X.shape
        k = pi.shape[0]
        g = np.zeros((k, n))
        for i in range(k):
            P = pdf(X, m[i], S[i])
            num = P * pi[i]
            g[i] = num
        den = np.sum(g, axis=0)
        g = g / den
        lg = np.sum(np.log(den))
        return g, lg
    except Exception:
        return None, None


def verify(X, pi, m, S):
    if not isinstance(X, np.ndarray):
        return False
    if len(X.shape) is not 2:
        return False
    if not isinstance(pi, np.ndarray):
        return False
    if len(pi.shape) is not 1:
        return False
    if not isinstance(m, np.ndarray):
        return False
    if len(m.shape) is not 2:
        return False
    if not isinstance(S, np.ndarray):
        return False
    if len(S.shape) is not 3:
        return False
    if pi.shape[0] is not m.shape[0]:
        return False
    if m.shape[1] is not X.shape[1]:
        return False
    if S.shape[0] is not pi.shape[0]:
        return False
    if S.shape[1] is not X.shape[1]:
        return False
    if S.shape[2] is not X.shape[1]:
        return False
    return True
