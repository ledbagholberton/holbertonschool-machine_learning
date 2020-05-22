#!/usr/bin/env python3
"""
X is a numpy.ndarray of shape (n, d) containing the data points
whose PDF should be evaluated
m is a numpy.ndarray of shape (d,) containing the mean of the distribution
S is a numpy.ndarray of shape (d, d) containing the covariance of the
distribution
You are not allowed to use any loops
Returns: P, or None on failure
P is a numpy.ndarray of shape (n,) containing the PDF values for
each data point
All values in P should have a minimum value of 1e-300
"""
import numpy as np


def pdf(X, m, S):
    """Calculates the PDF of a Gaussian distribution"""
    if not verify(X, m, S):
        return None
    n, d = X.shape
    S_det = np.linalg.det(S)
    S_inv = np.linalg.inv(S)
    N = np.sqrt((2*np.pi)**d * S_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    P = np.einsum('...k,kl,...l->...', X-m, S_inv, X-m)
    P = np.exp(-P / 2) / N
    P = np.clip(P, 1e-300, None)
    return P


def verify(X, m, S):
    """verifiy conditions"""
    if not isinstance(X, np.ndarray):
        return False
    if len(X.shape) is not 2:
        return False
    if not isinstance(m, np.ndarray):
        return False
    if not isinstance(S, np.ndarray):
        return False
    if m.shape[0] is not X.shape[1]:
        return False
    if len(m.shape) is not 1:
        return False
    if len(S.shape) is not 2:
        return False
    if S.shape[0] is not X.shape[1]:
        return False
    if S.shape[1] is not X.shape[1]:
        return False
    return True
