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
        A = np.matmul(self.W, X) + self.b
        H = self.sigmoid(A)
        self.__A = H
        return (self.__A)
