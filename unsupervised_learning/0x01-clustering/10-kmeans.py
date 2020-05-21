#!/usr/bin/env python3
"""
Write a function  that performs K-means on a dataset:

X is a numpy.ndarray of shape (n, d) containing the dataset
k is the number of clusters

Returns: C, clss
C is a numpy.ndarray of shape (k, d) containing
the centroid means for each cluster
clss is a numpy.ndarray of shape (n,) containing
the index of the cluster in C that each data point belongs to
"""
import sklearn.cluster


def kmeans(X, k):
    """K-means with scikit"""
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
