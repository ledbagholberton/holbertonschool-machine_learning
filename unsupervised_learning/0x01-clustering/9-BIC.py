"""
X is a numpy.ndarray of shape (n, d) containing the data set
"""
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
