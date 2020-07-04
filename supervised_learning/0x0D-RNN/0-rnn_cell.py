#!/usr/bin/env python3
"""
Create the class RNNCell that represents a cell of a simple RNN:

class constructor def __init__(self, i, h, o):
i is the dimensionality of the data
h is the dimensionality of the hidden state
o is the dimensionality of the outputs
Creates the public instance attributes Wh, Wy, bh, by that represent:
the weights and biases of the cell
Wh and bh are for the concatenated hidden state and input data
Wy and by are for the output
The weights should be initialized using:
a random normal distribution in the order listed above
The weights will be used on the right side for matrix multiplication
The biases should be initialized as zeros
"""
import numpy as np


class RNNCell:
    """Class RNNCell"""
    def __init__(self, i, h, o):
        """ Constructor
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ Method Forward
        performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains:
        the data input for the cell
        m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing:
        the previous hidden state
        Returns: h_next, y
        h_next is the next hidden state
        y is the output of the cell
        """
        m, i = x_t.shape
        _, h = h_prev.shape
        x_ht = np.hstack((h_prev, x_t))
        h_next = np.tanh(np.matmul(x_ht, self.Wh) + self.bh)
        y_n = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y_n)
        return (h_next, y)

    def softmax(self, X):
        expo = np.exp(X)
        expo_sum = np.sum(np.exp(X), axis=-1, keepdims=True)
        return expo/expo_sum
