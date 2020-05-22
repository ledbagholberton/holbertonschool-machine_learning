#!/usr/bin/env python3
"""
X is a numpy.ndarray of shape (n, d) containing the data set
k is a positive integer containing the number of clusters
iterations is a positive integer containing:
the maximum number of iterations for the algorithm
tol is a non-negative float:
containing tolerance of the log likelihood, used to determine early stopping
i.e. if the difference is less than or equal to tol you should stop
verbose is a boolean that determines if you should print information about
If True, print:
Log Likelihood after {i} iterations: {l}
every 10 iterations and after the last iteration
{i} is the number of iterations of the EM algorithm
{l} is the log likelihood
You may use at most 1 loop
Returns: pi, m, S, g, l, or None, None, None, None, None on failure
pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
m is a numpy.ndarray of shape (k, d) containing
the centroid means for each cluster
S is a numpy.ndarray of shape (k, d, d) containing
the covariance matrices for each cluster
g is a numpy.ndarray of shape (k, n) containing
the probabilities for each data point in each cluster
l is the log likelihood of the model
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """"performs the expectation maximization for a GMM"""
    try:
        if not verify(X, k, iterations, tol, verbose):
            return None, None, None, None, None
        n, d = X.shape
        pi, m, S = initialize(X, k)
        old_ll = 0
        for i in range(iterations):
            g, ll = expectation(X, pi, m, S)
            pi, m, S = maximization(X, g)
            if abs(old_ll - ll) < tol and verbose:
                print("Log Likelihood after {} iterations: {}"
                    .format(i, ll))
                break
            if ((i % 10 is 0) or (i is iterations - 1)) and verbose:
                print("Log Likelihood after {} iterations: {}"
                    .format(i, ll))
            old_ll = ll
        return(pi, m, S, g, ll)
    except Exception:
        return None, None, None, None, None


def verify(X, k, iterations, tol, verbose):
    """verifiy conditions"""
    if not isinstance(X, np.ndarray):
        return False
    if len(X.shape) is not 2:
        return False
    if type(k) is not int or k <= 0 or X.shape[0] <= k:
        return False
    if type(iterations) is not int or iterations <= 0:
        return False
    if type(tol) is not float or tol < 0:
        return False
    if type(verbose) is not bool:
        return False
    return True
