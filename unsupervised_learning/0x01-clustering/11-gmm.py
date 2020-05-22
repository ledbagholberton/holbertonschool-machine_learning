#!/usr/bin/env python3
"""
Write a function  that calculates a GMM from a dataset:

X is a numpy.ndarray of shape (n, d) containing the dataset
k is the number of clusters
Returns: pi, m, S, clss, bic
pi is a numpy.ndarray of shape (k,) containing the cluster priors
m is a numpy.ndarray of shape (k, d) containing the centroid means
S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
clss is a numpy.ndarray of shape (n,) containing cluster indices for each point
bic is a numpy.ndarray of shape (kmax - kmin + 1) containing:
BIC value for each cluster size tested
"""
import sklearn.mixture


def gmm(X, k):
    """GMM with scikit"""
    gmm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic
