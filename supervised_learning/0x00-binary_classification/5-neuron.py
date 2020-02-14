#!/usr/bin/env python3
""" Neuron class: defines a single neuron performing binary classification
with pivate instances attributes
"""

import numpy as np


class Neuron:
    """ Class Neuron """
    def __init__(self, nx):
        """ Settings for class Neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def sigmoid(self, Z):
        sigm = 1 / (1 + np.exp(-Z))
        return sigm

    def forward_prop(self, X):
        recta = np.matmul(self.W, X) + self.b
        H = self.sigmoid(recta)
        self.__A = H
        return (self.__A)

    def log_reg(self, y_r, y_p):
        num_lreg = -1 * (y_r * np.log(y_p) + (1 - y_r) *
                         np.log(1.0000001 - y_p))
        return (num_lreg)

    def cost(self, Y, A):
        m = Y.shape[1]
        num_lreg = -1 * (Y * np.log(A) + (1 - Y) *
                         np.log(1.0000001 - A))
        cost = np.sum(num_lreg)/m
        return (cost)

    def evaluate(self, X, Y):
        self.forward_prop(X)
        PRED = np.where(self.__A >= 0.5, 1, 0)
        return (PRED, self.cost(Y, self.__A))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        m = Y.shape[1]
        Z = A - Y
        db = np.sum(Z) / m
        self.__b = self.__b - alpha * db
        ZT = Z.T
        dW = np.matmul(X, ZT) / m
        self.__W = self.__W - (alpha * dW.T)
        return()
