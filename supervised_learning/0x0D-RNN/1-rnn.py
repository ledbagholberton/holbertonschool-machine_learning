#!/usr/bin/env python3
"""
Function Forward propagation

"""
import numpy as np

    
def rnn(rnn_cell, X, h_0):
    """performs forward propagation for a simple RNN
    rnn_cell is an instance of RNNCell that will be used for the
    forward propagation
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    h_0 is the initial hidden state, given as numpy.ndarray of shape (m, h)
    h is the dimensionality of the hidden state
    Returns: H, Y
    H is a numpy.ndarray containing all of the hidden states
    Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape
    H = h_0
    H = np.reshape(H, (1, m, h))
   
    for iter in range(t):
        h_next, y = rnn_cell.forward(h_0, X[iter, :, :])
        h_next_r = np.reshape(h_next, (1, m, h))
        H = np.vstack((H, h_next_r))
        if iter == 0:
            yv = y.shape[1]
            y_r = np.reshape(y, (1, m, yv))
            Y = y_r
        else:
            y_r = np.reshape(y, (1, m, yv))
            Y = np.vstack((Y, y_r))
        h_0 = h_next
    return(H, Y)
