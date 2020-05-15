#!/usr/bin/env python3
"""
Write a function  that performs a t-SNE transformation:

X is a numpy.ndarray of shape (n, d) containing the dataset to be
transformed by t-SNE
n is the number of data points
d is the number of dimensions in each point
ndims is the new dimensional representation of X
idims is the intermediate dimensional representation of X after PCA
perplexity is the perplexity
iterations is the number of iterations
lr is the learning rate
Returns: Y, a numpy.ndarray of shape (n, ndim) containing the optimized
low dimensional transformation of X
You should use:
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost
For the first 100 iterations, perform early exaggeration with
an exaggeration of 4
a(t) = 0.5 for the first 20 iterations and 0.8 thereafter
"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """Function tsne"""
    n, d = X.shape
    Y = np.random.randn(n, ndims)
    iY = np.zeros((n, ndims))
    gains = np.ones((n, ndims))
    X = pca(X, idims)
    P = P_affinities(X, perplexity=perplexity)
    # Early exaggeration
    P = P * 4
    a = 0.5
    Yt1 = Yt2 = 0
    min_gain = 0.01
    for i in range(iterations):
        if i is 20:
            a = 0.8
        if i is 100:
            P = P / 4
        dY, Q = grads(Y, P)
        C = cost(P, Q)
        if (i + 1) % 100 is 0:
            print("Cost at iteration {}: {}".format(i+1, C))
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = a * iY - lr * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        """Y = Yt1 + lr*dY + a * (Yt1 - Yt2)
        Yt1 = Y
        Yt2 = Yt1"""
    return Y
