#!/usr/bin/env python3
"""
Function Forward propagation

"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    performs forward propagation for a bidirectional RNN:
    bi_cells is an instance of BidirectinalCell that will be used
    for the forward propagation
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    h_0 is the initial hidden state in the forward direction,
    given as a numpy.ndarray of shape (m, h)
    h is the dimensionality of the hidden state
    h_t is the initial hidden state in the backward direction,
    given as a numpy.ndarray of shape (m, h)
    Returns: H, Y
    H is a numpy.ndarray containing all of the concatenated hidden states
    Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape
    h_t_0 = np.zeros_like(h_0)
    H = np.zeros((t, m, 2*h))
    for iter in range(t):
        if iter == 0:
            h_next = bi_cell.forward(h_0, X[iter])
            h_prev = bi_cell.backward(h_t, X[iter])
        elif iter == t - 1:
            h_next = bi_cell.forward(H[iter-1, :, 0:15], X[iter])
            h_prev = bi_cell.backward(h_t_0, X[iter])
        else:
            h_next = bi_cell.forward(H[iter-1, :, 0:15], X[iter])
            h_prev = bi_cell.backward(H[iter+1, :, 15:30], X[iter])
        H[iter] = np.concatenate((h_next, h_prev), axis=1)
    Y = bi_cell.output(H)
    return(H, Y)
