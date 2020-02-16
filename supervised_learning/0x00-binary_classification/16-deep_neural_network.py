#!/usr/bin/env python3
""" Deep Neural Network class: defines a deep neural network class
with private instances attributes
"""

import numpy as np
import matplotlib.pyplot as plt


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
        if not all(x > 0 and type(x) is int for x in layers):
            raise ValueError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = he_et_al(nx, layers)


def he_et_al(nx, ly):
    """
    Arguments:
    layers: list containing the numbers of Layers and the number of nodes on
    each layer.
    Initialization of each weight is based on method from He.
    Returns:
    weights -- python dict containing your parameters "W1","b1".. "WL", "bL":
                W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                b1 -- bias vector of shape (layers_dims[1], 1)
                ...
                WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                bL -- bias vector of shape (layers_dims[L], 1)
    """

    wg = {}
    myL = len(ly) + 1
    ly.insert(0, nx)
    for l in range(1, myL):
        wg['W' + str(l)] = np.random.randn(ly[l], ly[l-1])*(np.sqrt(2/ly[l-1]))
        wg['b' + str(l)] = np.zeros((ly[l], 1))
    return wg
