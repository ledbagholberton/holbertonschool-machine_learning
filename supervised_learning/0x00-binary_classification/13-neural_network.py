#!/usr/bin/env python3
""" Neural Network class: defines a neural network class
with pivate instances attributes
"""

import numpy as np


class NeuralNetwork:
    """ Class Neural networks"""
    def __init__(self, nx, nodes):
        """ Settings for class Neural Networks"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def sigmoid(self, Z):
        """Function Sigmoid"""
        sigm = 1 / (1 + np.exp(-Z))
        return sigm

    def forward_prop(self, X):
        """Function Forward Propagation"""
        A1 = np.matmul(self.__W1, X) + self.__b1
        H1 = self.sigmoid(A1)
        self.__A1 = H1
        A2 = np.matmul(self.__W2, self.__A1) + self.__b2
        H2 = self.sigmoid(A2)
        self.__A2 = H2
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """Function Cost"""
        m = Y.shape[1]
        num_lreg = -1 * (Y * np.log(A) + (1 - Y) *
                         np.log(1.0000001 - A))
        cost = np.sum(num_lreg)/m
        return (cost)

    def evaluate(self, X, Y):
        """Function evauate"""
        self.forward_prop(X)
        PRED = np.where(self.__A2 >= 0.5, 1, 0)
        return (PRED, self.cost(Y, self.__A2))

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Function gradient descent"""
        m2 = A1.shape[1]
        dZ2 = A2 - Y
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m2
        dZT2 = dZ2.T
        dW2 = np.matmul(A1, dZT2) / m2
        m1 = Y.shape[1]
        dg1 = (A1 * (1 - A1))
        dZ1 = np.matmul(self.__W2.T, dZ2) * dg1
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m1
        dZT1 = dZ1.T
        dW1 = np.matmul(X, dZT1) / m1
        self.__W1 = self.__W1 - (alpha * dW1.T)
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - (alpha * dW2.T)
        self.__b2 = self.__b2 - alpha * db2
        return()
