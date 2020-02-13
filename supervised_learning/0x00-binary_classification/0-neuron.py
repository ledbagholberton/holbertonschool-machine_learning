#!/usr/bin/env python3
""" Neuron class: defines a single neuron performing binary classification
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
        self.W = np.random.randn(nx)
        self.b = 0
        self.A = 0
