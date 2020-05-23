#!/usr/bin/env python3
"""
Write a function  that :

X is a numpy.ndarray of shape (n, d) containing the data set
g is a numpy.ndarray of shape (k, n) containing the posterior
probabilities for each data point in each cluster
You may use at most 1 loop
Returns: pi, m, S, or None, None, None on failure
pi is a numpy.ndarray of shape (k,) containing the updated priors
for each cluster
m is a numpy.ndarray of shape (k, d) containing the updated centroid means
for each cluster
S is a numpy.ndarray of shape (k, d, d) containing the updated covariance
matrices for each cluster
"""
import numpy as np


def maximization(X, g):
    """"calculates the maximization step in the EM algorithm for a GMM"""
    if not verify(X, g):
        return None, None, None
    n, d = X.shape
    k, _ = g.shape
    m = np.zeros((k, d))
    S = np.empty((k, d, d))
    pi = np.zeros((k, ))
    for i in range(k):
        Nk = np.sum(g[i])
        pi[i] = Nk / n
        gi = g[i].reshape(1, n)
        m[i] = np.sum(np.matmul(gi, X), axis=0) / Nk
        Df = X - m[i]
        S[i] = np.dot(gi * Df.T, Df) / Nk
    return(pi, m, S)


def verify(X, g):
    """verifiy conditions"""
    if not isinstance(X, np.ndarray):
        return False
    if len(X.shape) is not 2:
        return False
    if not isinstance(g, np.ndarray):
        return False
    if len(g.shape) is not 2:
        return False
    if X.shape[0] != g.shape[1]:
        return False
    sum_g = np.sum(g, axis=0)
    if not np.allclose(sum_g, np.ones_like(g)):
        return False
    return True
