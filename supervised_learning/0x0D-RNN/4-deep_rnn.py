#!/usr/bin/env python3
"""
Function Forward propagation

"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """performs forward propagation for a Deep RNN
    rnn_cells is a list of RNNCell instances of length l
    that will be used for the forward propagation
    l is the number of layers
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    h_0 the initial hidden state, given as a numpy.ndarray of shape (l, m, h)
    h is the dimensionality of the hidden state
    Returns: H, Y
    H is a numpy.ndarray containing all of the hidden states
    Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    l, _, h = h_0.shape
    H = np.zeros((t+1, l, m, h))
    H[0] = h_0
    for iter in range(1, t + 1):
        h_prev = h_0
        for layers in range(l):
            if layers == 0:
                h_next, y = rnn_cells[layers].forward(H[iter-1,
                                                        layers], X[iter-1])
            else:
                h_next, y = rnn_cells[layers].forward(H[iter-1,
                                                        layers], h_prev)
            h_prev = h_next
            H[iter, layers] = h_next
        if iter-1 == 0:
            yv = y.shape[1]
            y_r = np.reshape(y, (1, m, yv))
            Y = y_r
        else:
            y_r = np.reshape(y, (1, m, yv))
            Y = np.vstack((Y, y_r))
    return(H, Y)
