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

    def backward(self, h_next, x_t):
        """ Method backward
        public instance method  that calculates the hidden state in the
        backward direction for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains:
            the data input for the cell m is the batch size for the data
        h_next is a numpy.ndarray of shape (m, h) containing:
            the next hidden state
        Returns: h_prev, the previous hidden state
        """
        m, i = x_t.shape
        _, h = h_next.shape
        x_ht = np.hstack((h_next, x_t))
        h_next = np.tanh(np.matmul(x_ht, self.Whb) + self.bhb)
        return (h_next)

    def output(self, H):
        """ Method output that calculates all outputs for the RNN:
        H is a numpy.ndarray of shape (t, m, 2 * h) that contains:
            the concatenated hidden states from both directions, excluding:
            their initialized states
        t is the number of time steps
        m is the batch size for the data
        h is the dimensionality of the hidden states
        Returns: Y, the outputs
        """
        Y_n = np.matmul(H, self.Wy) + self.by
        Y = self.softmax(Y_n)
        return Y

    def softmax(self, X):
        """Exponential function"""
        expo = np.exp(X)
        expo_sum = np.sum(np.exp(X), axis=-1, keepdims=True)
        return expo/expo_sum

    def sigmoid(self, X):
        """Sigmoid fucntion"""
        return (1/(1 + np.exp(-X)))
