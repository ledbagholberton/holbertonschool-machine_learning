#!/usr/bin/env python3
"""Write a function  that initializes all variables required
to calculate the P affinities in t-SNE:

X is a numpy.ndarray of shape (n, d) containing the dataset
to be transformed by t-SNE
n is the number of data points
d is the number of dimensions in each point
perplexity is the perplexity that all Gaussian distributions should have
Returns: (D, P, betas, H)
D: a numpy.ndarray of shape (n, n) that calculates the pairwise
distance between two data points
P: a numpy.ndarray of shape (n, n) initialized to all 0‘s that
will contain the P affinities
betas: a numpy.ndarray of shape (n, 1) initialized to all 1’s
that will contain all of the beta values
H is the Shannon entropy for perplexity perplexity"""
import numpy as np


def P_init(X, perplexity):
    """FUnction Init """
    n, d = X.shape
    sum_X = np.sum(np.square(X), 1, keepdims=True)
    D = np.add(np.add(-2 * np.matmul(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H
