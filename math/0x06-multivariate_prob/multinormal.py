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
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        elif len(data.shape) is not 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        elif data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        else:
            X = data.T
            d = data.shape[0]
            mean = np.sum(X, axis=0)/X.shape[0]
            self.mean = np.mean(data, axis=1).reshape(d, 1)
            a = X - mean
            b = a.T
            c = np.matmul(b, a)
            self.cov = c/(X.shape[0] - 1)

    def pdf(self, x):
        """Function calculate PDF
        Probability Density Function
        x is a numpy.ndarray of shape (d, 1) containing the data point whose
        PDF should be calculated
        d is the number of dimensions of the Multinomial instance
        If x is not a numpy.ndarray, raise a TypeError with the message
        x must by a numpy.ndarray
        If x is not of shape (d, 1), raise a ValueError with the message
        x mush have the shape ({d}, 1)
        Returns the value of the PDF"""
        if not(isinstance(x, np.ndarray)):
            raise TypeError("x must be a numpy.ndarray")
        elif (x.shape is not 2 and x.shape[1] is not 1):
            raise TypeError("x mush have the shape ({d}, 1)".
                            format(x.shape[0]))
        else:
            cov = self.cov
            inv_cov = np.linalg.inv(cov)
            mean = self.mean
            D = x.shape[0]
            det_cov = np.linalg.det(cov)
            den = np.sqrt(np.power((2 * np.pi), D) * det_cov)
            y = np.matmul((x - mean).T, inv_cov)
            pdf = (1 / den) * np.exp(-1 * np.matmul(y, (x - mean)) / 2)
            return pdf.reshape(-1)[0]
