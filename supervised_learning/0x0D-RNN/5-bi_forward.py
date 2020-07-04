#!/usr/bin/env python3
"""
Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by that
represent the weights and biases of the cell
Whf and bhfare for the hidden states in the forward direction
Whb and bhbare for the hidden states in the backward direction
Wy and byare for the outputs
The weights should be initialized using a random normal distribution
in the order listed above
The weights will be used on the right side for matrix multiplication
The biases should be initialized as zeros

"""
import numpy as np


class BidirectionalCell:
    """Class Bidirectional"""
    def __init__(self, i, h, o):
        """ Constructor
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Whf = np.random.randn(h + i, h)
        self.Whb = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h + h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ Method Forward
        calculates the hidden state in the forward direction for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains:
            the data input for thecell
        m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing:
            the previous hidden state
        Returns: h_next, the next hidden state
        """
        m, i = x_t.shape
        _, h = h_prev.shape
        x_ht = np.hstack((h_prev, x_t))
        h_next = np.tanh(np.matmul(x_ht, self.Whf) + self.bhf)
        return (h_next)
