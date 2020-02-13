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
        np.random.seed(1)

    @property
    def W(self, nx):
        self.__W = np.random.randn(self.nx)
        return self.__W

    @property
    def b(self):
        self.__b = 0
        return self.__b

    @property
    def A(self):
        self.__A = 0
        return self.__A
