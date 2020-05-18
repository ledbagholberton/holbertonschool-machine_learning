#!/usr/bin/env python3
""" t-SNE """
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    performs a t-SNE transformation:
        - X is a numpy.ndarray of shape (n, d) containing the dataset
            to be transformed by t-SNE
            - n is the number of data points
            - d is the number of dimensions in each point
        - ndims is the new dimensional representation of X
        - idims is the intermediate dimensional representation of X after PCA
        - perplexity is the perplexity
        - iterations is the number of iterations
        - lr is the learning rate
        Returns: Y, a numpy.ndarray of shape (n, ndim) containing the
            optimized low dimensional transformation of X
        For the first 100 iterations, perform early exaggeration with an
        exaggeration of 4
        a(t) = 0.5 for the first 20 iterations and 0.8 thereafter
    """
    X = pca(X, idims)
    n, _ = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    P = P_affinities(X, perplexity=perplexity) * 4
    Y = np.random.randn(n, ndims)
    iY = np.zeros((n, ndims))
    for i in range(iterations):
        dY, Q = grads(Y, P)
        if i < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        iY = momentum * iY - lr * dY
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        if i != 0 and i % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i, C))
        if i == 100:
            P = P / 4.
    return Y