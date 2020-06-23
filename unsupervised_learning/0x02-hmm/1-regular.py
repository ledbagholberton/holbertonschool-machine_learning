#!/usr/bin/env python3
"""
P is a is a square 2D numpy.ndarray of shape (n, n) representing
the transition matrix
P[i, j] is the probability of transitioning from state i to state j
n is the number of states in the markov chain
Returns: a numpy.ndarray of shape (1, n) containing
the steady state probabilities, or None on failure
"""
import numpy as np


def regular(P):
    """
    determines the steady state probabilities of a regular markov chain
    """
    try:
        if (not isinstance(P, np.ndarray)):
            return None
        if len(P.shape) != 2:
            return None
        if P.shape[0] is not P.shape[1]:
            return None
        a = np.sum(P) / P.shape[0]
        if a != 1:
            return None
        n = P.shape[0]
        S = np.ones((1, n))/n
        Pj = P.copy()
        while True:
            Multi = S
            S = np.matmul(S, P)
            Pj = P * Pj
            if np.any(Pj <= 0):
                return None
            if np.all(Multi == S):
                break
        return S
    except Exception:
        return None
