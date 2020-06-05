#!/usr/bin/env python3
"""
Sets the public instance attributes X, Y, l, and sigma_f corresponding to the
respective constructor inputs
Sets the public instance attribute K, representing the current covariance
kernel matrix for the Gaussian process
Public instance method def kernel(self, X1, X2): that calculates the
covariance kernel matrix between two matrices:

X1 is a numpy.ndarray of shape (m, 1)
X2 is a numpy.ndarray of shape (n, 1)
the kernel should use the Radial Basis Function (RBF)
Returns: the covariance kernel matrix as a numpy.ndarray of shape (m, n)
"""


class GaussianProcess:
    """Represent a Noiseless 1D Gaussian Process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Class constructor
        X_init is a numpy.ndarray of shape (t, 1) representing
        the inputs already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing
        the outputs of the black-box function for each input in X_init
        t is the number of initial samples
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of
        the black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        import numpy as np

        """calculates the covariance kernel matrix between two matrices
        X1 is a numpy.ndarray of shape (m, 1)
        X2 is a numpy.ndarray of shape (n, 1)
        the kernel should use the Radial Basis Function (RBF)
        Return: the covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        # sqdist = np.linalg.norm(np.subtract(X1, X2))
        sqdist = (np.sum(X1**2, 1).reshape(-1, 1)
                  + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T))
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        import numpy as np

        """
        predicts the mean and standard deviation of points in Gaussian process
        X_s is a numpy.ndarray of shape (s, 1) containing
        all the points whose mean and standard deviation should be calculated
        s is the number of sample points
        Returns: mu, sigma
        mu is a numpy.ndarray of shape (s,) containing
        the mean for each point in X_s, respectively
        sigma is a numpy.ndarray of shape (s,) containing
        the standard deviation for each point in X_s, respectively
        """
        s = X_s.shape[0]
        K = self.kernel(self.X, self.X)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        # Equation (4)
        mu = K_s.T.dot(K_inv).dot(self.Y)
        mu = np.reshape(mu, (s,))
        # Equation (5)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov_s)
        return mu, sigma

    def update(self, X_new, Y_new):
        import numpy as np

        """
        X_new is a numpy.ndarray shape (1,) represents the new sample point
        Y_new is numpy.ndarray shape (1,) represents new sample function value
        Updates the public instance attributes X, Y, and K
        """
        X_new = X_new.reshape((1, 1))
        Y_new = Y_new.reshape((1, 1))
        self.X = np.concatenate((self.X, X_new), axis=0)
        self.Y = np.concatenate((self.Y, Y_new), axis=0)
        self.K = self.kernel(self.X, self.X)
