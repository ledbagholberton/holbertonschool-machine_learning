#!/usr/bin/env python3
"""
P is a is a square 2D numpy.ndarray of shape (n, n) representing
the transition matrix
P[i, j] is the probability of transitioning from state i to state j
n is the number of states in the markov chain
Returns: True if it is absorbing, or False on failure
"""
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing"""
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
        lista = np.zeros((1, n))
        if (P == np.eye(P.shape[0])).all():
            return True
        for i in range(n):
            if P[i][i] == 1:
                lista[0][i] = 1
                for j in range(n):
                    if P[j][i] != 0:
                        lista[0][j] += 1
        for i in range(n):
            if lista[0][i] != 0:
                for j in range(n):
                    if P[j][i] != 0:
                        lista[0][j] += 1
        if np.any(lista == 0):
            return False
        return True
        idx = np.argsort(lista, axis=-1)
        P = P[idx]
    except Exception:
        return False
