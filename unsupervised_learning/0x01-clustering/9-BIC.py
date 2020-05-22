#!/usr/bin/env python3
"""
X is a numpy.ndarray of shape (n, d) containing the data set
kmin is a positive integer containing
the minimum number of clusters to check for (inclusive)
kmax is a positive integer containing
the maximum number of clusters to check for (inclusive)
iterations is a positive integer containing
the maximum number of iterations for the EM algorithm
tol is a non-negative float containing the tolerance for the EM algorithm
verbose is a boolean that determines if the EM algorithm should print
information to the standard output
Returns:
best_k is the best value for k based on its BIC
best_result is tuple containing pi, m, S
pi is a numpy.ndarray of shape (k,)
containing the cluster priors for the best number of clusters
m is a numpy.ndarray of shape (k, d)
containing the centroid means for the best number of clusters
S is a numpy.ndarray of shape (k, d, d)
containing the covariance matrices for the best number of clusters
l is a numpy.ndarray of shape (kmax - kmin + 1)
containing the log likelihood for each cluster size tested
b is a numpy.ndarray of shape (kmax - kmin + 1)
containing the BIC value for each cluster size tested
Use: BIC = p * ln(n) - 2 * l
p is the number of parameters required for the model
n is the number of data points used to create the model
l is the log likelihood of the model"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """finds the best number of clusters for a GMM using }
    the Bayesian Information Criterion"""
    if not verify(X, kmin, kmax, iterations, tol, verbose):
        return None, None, None, None
    n, d = X.shape
    old_BIC = 1e10
    b = np.zeros((kmax - kmin + 1))
    logs = np.zeros((kmax - kmin + 1)).astype(float)
    for k in range(kmin, kmax + 1, 1):
        pi, m, S, g, ll = expectation_maximization(X, k, iterations, tol,
                                                   verbose)
        p = (k * d * (d+1) / 2) + (d * k) + k - 1
        BIC = p * np.log(n) - 2 * ll
        b[k-kmin] = BIC
        logs[k-kmin] = ll
        if BIC < old_BIC:
            best_k = k
            best_result = (pi, m, S)
        old_BIC = BIC
    return(best_k, best_result, logs, b)


def verify(X, kmin, kmax, iterations, tol, verbose):
    """Verify requirements"""
    if not isinstance(X, np.ndarray):
        return False
    if len(X.shape) is not 2:
        return False
    if type(kmin) is not int or kmin <= 0 or X.shape[0] <= kmin:
        return False
    if type(kmax) is not int or kmax <= 0 or X.shape[0] <= kmax:
        return False
    if kmax <= kmin:
        return False
    if type(iterations) is not int or iterations <= 0:
        return False
    if type(tol) is not float or tol < 0:
        return False
    if type(verbose) is not bool:
        return False
    return True
