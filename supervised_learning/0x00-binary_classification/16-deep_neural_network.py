#!/usr/bin/env python3
""" Deep Neural Network class: defines a deep neural network class
with private instances attributes
"""

import numpy as np


class DeepNeuralNetwork:
    """ Class Deep Neural networks"""
    def __init__(self, nx, layers):
        """ Settings for class Deep Neural Networks"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        ly = layers.copy()
        ly.insert(0, nx)
        if (type(ly[self.L + 1]) != int or ly[self.L + 1] <= 0):
            raise TypeError("layers must be a list of positive integers")
        for l in range(1, self.L + 1):
            if (type(ly[l-1]) != int or ly[(l-1)] <= 0):
                raise TypeError("layers must be a list of positive integers")
            else:
                temp = np.random.randn(ly[l], ly[l-1]) * (np.sqrt(2/ly[l-1]))
                self.weights['W'+str(l)] = temp
                self.weights['b'+str(l)] = np.zeros((ly[l], 1))
