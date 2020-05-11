#!/usr/bin/env python3
"""
data is a numpy.ndarray of shape (d, n) containing the data set:
n is the number of data points
d is the number of dimensions in each data point
If data is not a 2D numpy.ndarray, raise a TypeError with the message
data must be a 2D numpy.ndarray
If n is less than 2, raise a ValueError with the message data must
contain multiple data points
Set the public instance variables:
mean - a numpy.ndarray of shape (d, 1) containing the mean of data
cov - a numpy.ndarray of shape (d, d) containing the covariance
matrix data
You are not allowed to use the function numpy.cov"""
import numpy as np


class MultiNormal:
    """Class Multivariate Normal distribution"""

    def __init__(self, data):
        "Constructor"
        a = len(data.shape)
        if len(data.shape) is not 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        elif data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        media, self.cov = self.mean_cov(data)
        self.mean = media.T

    def mean_cov(self, X):
        """FUnction calculate mean and covariance"""
        mean = np.sum(X, axis=1)/X.shape[1]
        a = X.T - mean
        b = a.T
        c = np.matmul(b, a)
        cov = c/(X.shape[1] - 1)
        return(mean, cov)

    def correlation(self, C):
        """Function correlation"""
        v = np.sqrt(np.diag(C))
        outer_v = np.outer(v, v)
        correlation = C / outer_v
        correlation[C == 0] = 0
        return correlation
