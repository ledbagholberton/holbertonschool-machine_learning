#!/usr/bin/env python3
"""
calculates the symmetric P affinities of a data set

X is a numpy.ndarray of shape (n, d)
containing the dataset to be transformed by t-SNE
n is the number of data points
d is the number of dimensions in each point
perplexity is the perplexity that all Gaussian distributions should have
tol is the maximum tolerance allowed (inclusive) for the difference
in Shannon entropy from perplexity for all Gaussian distributions

Returns: P, a numpy.ndarray of shape (n, n)
containing the symmetric P affinities
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """P_affinities"""
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)
    # with the function entropy (HP) we obtain the
    # P-affinities at specific point and the Shannon Entropy at
    # this point. Now we need to search

    for iter in range(n):
        beta_min, beta_max = None, None
        D_iter = np.delete(D[iter], iter, axis=0)
        H_iter, P_iter = HP(D_iter, betas[iter, 0])
        # lo que toca encontrar es cual beta cumple para que la
        # perplejidad dada y la encontrada tengan una diferencia max de tol
        dif_H = H_iter - H
        while np.abs(dif_H) > tol:
            if dif_H > 0:
                beta_min = betas[iter][0]
                if beta_max is None:
                    betas[iter, 0] = betas[iter, 0] * 2
                else:
                    betas[iter, 0] = (betas[iter, 0] + beta_max)/2
            else:
                beta_max = betas[iter, 0]
                if beta_min is None:
                    betas[iter, 0] = betas[iter, 0] / 2
                else:
                    betas[iter, 0] = (betas[iter, 0] + beta_min)/2
            H_iter, P_iter = HP(D_iter, betas[iter, 0])
            dif_H = H_iter - H
        P_iter = np.insert(P_iter, iter, 0, axis=0)
        P[iter] = P_iter
    P = (P + P.T) / (2*n)
    return(P)
