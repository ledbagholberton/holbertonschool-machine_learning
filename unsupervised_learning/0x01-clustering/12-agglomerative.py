#!/usr/bin/env python3
"""
Write a function  that performs agglomerative clustering on a dataset:

X is a numpy.ndarray of shape (n, d) containing the dataset
dist is the maximum cophenetic distance for all clusters
Performs agglomerative clustering with Ward linkage
Displays the dendrogram with each cluster displayed in a different color
Returns: clss, a numpy.ndarray of shape (n,) containing
the cluster indices for each data point
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


def agglomerative(X, dist):
    model = AgglomerativeClustering(distance_threshold=dist,
                                    n_clusters=None,
                                    linkage='ward')
    model = model.fit(X)
    dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    return model.labels_
