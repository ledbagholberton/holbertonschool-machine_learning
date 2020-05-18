#!/usr/bin/env python3
""" P affinities """
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    calculates the symmetric P affinities of a data set:
        - X is a numpy.ndarray of shape (n, d) containing the dataset
            to be transformed by t-SNE
            - n is the number of data points
            - d is the number of dimensions in each point
        - perplexity is the perplexity that all Gaussian
            distributions should have
        - tol is the maximum tolerance allowed (inclusive) for the
            difference in Shannon entropy from perplexity for all
            Gaussian distributions
        Returns: P, a numpy.ndarray of shape (n, n) containing the
        symmetric P affinities
    """
    n, _ = X.shape
    D, P, betas, H = P_init(X, perplexity)
    for i in range(n):
        high = None
        low = None
        Di = np.delete(D[i], i, axis=0)
        Hi, Pi = HP(Di, betas[i])
        diff = Hi - H
        # binary search
        while np.abs(diff) > tol:
            if diff > 0:
                low = betas[i, 0]
                if high is None:
                    betas[i, 0] = betas[i, 0] * 2
                else:
                    betas[i, 0] = (betas[i, 0] + high) / 2
            else:
                high = betas[i, 0]
                if low is None:
                    betas[i, 0] = betas[i, 0] / 2
                else:
                    betas[i, 0] = (betas[i, 0] + low) / 2
            Hi, Pi = HP(Di, betas[i])
            diff = Hi - H
        Pi = np.insert(Pi, i, 0)
        P[i] = Pi
    # simmetric
    P = (P + P.T)/(2*n)
    return(P)