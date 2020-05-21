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
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


def agglomerative(X, dist):
    model = AgglomerativeClustering(distance_threshold=dist,
                                    n_clusters=None,
                                    linkage='ward')
    model = model.fit(X)
    plt.title('Hierarchical Clustering Dendrogram')
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    return model.labels_
