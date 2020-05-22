#!/usr/bin/env python3
"""
Write a function  that performs K-means on a dataset:

X is a numpy.ndarray of shape (n, d) containing the dataset
n is the number of data points
d is the number of dimensions for each data point
k is a positive integer containing the number of clusters
iterations is a positive integer containing the maximum number
of iterations that should be performed
If no change occurs between iterations, your function should return
Initialize the cluster centroids using a multivariate uniform
distribution (based on0-initialize.py)
If a cluster contains no data points during the update step, reinitialize
its centroid
You should use numpy.random.uniform exactly twice
You may use at most 2 loops
Returns: C, clss, or None, None on failure
C is a numpy.ndarray of shape (k, d) containing the centroid means
for each cluster
clss is a numpy.ndarray of shape (n,) containing the index of the cluster
in C that each data point belongs to
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset"""
    try:
        if not verify(X, k, iterations):
            return None, None
        n, d = X.shape
        low = np.amin(X, axis=0)
        high = np.amax(X, axis=0)
        centroids = np.random.uniform(low=low, high=high, size=(k, d))
        old_centroids = np.zeros((k, d))
        for iter in range(iterations):
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2)
                                .sum(axis=-1))
            closest = np.argmin(distances, axis=0)
            lista = []
            for i in range(k):
                a = X[closest == i]
                if len(a) == 0:
                    media = np.random.uniform(low=low, high=high, size=(d))
                else:
                    media = np.mean(a, axis=0)
                lista.append(media)
            centroids = np.array(lista)
            if np.array_equal(old_centroids, centroids):
                break
            old_centroids = centroids
        return(centroids, closest)
    except BaseException:
        return None, None


def verify(X, k, iterations):
    """verifiy conditions"""
    if not isinstance(X, np.ndarray):
        return False
    if len(X.shape) is not 2:
        return False
    if type(k) is not int or k <= 0 or k >= X.shape[0]:
        return False
    if type(iterations) is not int or iterations <= 0:
        return False
    return True
