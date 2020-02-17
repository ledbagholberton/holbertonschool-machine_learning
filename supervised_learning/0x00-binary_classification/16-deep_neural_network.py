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
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        """if not all(x > 0 and type(x) is int for x in layers):
            raise ValueError("layers must be a list of positive integers")"""
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        myL = self.L + 1
        ly = layers.copy()
        ly.insert(0, nx)
        for l in range(1, myL):
            temp = np.random.randn(ly[l], ly[l-1]) * (np.sqrt(2/ly[l-1]))
            self.weights['W'+str(l)] = temp
            self.weights['b'+str(l)] = np.zeros((ly[l], 1))
