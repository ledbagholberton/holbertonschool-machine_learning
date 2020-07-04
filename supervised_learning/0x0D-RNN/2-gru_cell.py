#!/usr/bin/env python3
"""
Create the class GRUCell that represents a gated recurrent unit:

class constructor def __init__(self, i, h, o):
i is the dimensionality of the data
h is the dimensionality of the hidden state
o is the dimensionality of the outputs
Creates the public instance attributes:
Wz, Wr, Wh, Wy, bz, br, bh, by that represent weights and biases of the cell
Wzand bz are for the update gate
Wrand br are for the reset gate
Whand bh are for the intermediate hidden state
Wyand by are for the output
The weights should be initialized using:
    a random normal distribution in the order listed above
The weights will be used on the right side for matrix multiplication
The biases should be initialized as zeros
"""
import numpy as np


class GRUCell:
    """Class RNNCell"""
    def __init__(self, i, h, o):
        """ Constructor
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ Method Forward
        public instance method that performs forward propagation
        for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains
        the data input for the cell
        m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing
        the previous hidden state
        Returns: h_next, y
        h_next is the next hidden state
        y is the output of the cell
        """
        m, i = x_t.shape
        _, h = h_prev.shape
        st_ct_1 = np.hstack((h_prev, x_t))
        g_u = self.sigmoid(np.matmul(st_ct_1, self.Wz) + self.bz)
        g_r = self.sigmoid(np.matmul(st_ct_1, self.Wr) + self.br)
        st_ct_full = np.hstack(((g_r * h_prev), x_t))
        c_tilde = np.tanh(np.matmul(st_ct_full, self.Wh) + self.bh)
        h_next = (g_u * c_tilde) + ((1-g_u) * h_prev)
        y_n = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y_n)
        return (h_next, y)

    def softmax(self, X):
        """Softmax Function"""
        np.exp(X)
        expo_sum = np.sum(np.exp(X), axis=-1, keepdims=True)
        return expo/expo_sum

    def sigmoid(self, X):
        """Sigmoid function"""
        return 1/(1 + np.exp(-X))
