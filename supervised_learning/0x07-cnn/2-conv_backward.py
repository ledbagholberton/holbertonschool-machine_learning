#!/usr/bin/env python3
""" Function Conv Backward
Write a function that performs back propagation over a convolutional
layer of a neural network:

dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
partial derivatives with respect to the unactivated output of the convolutional
layer
m is the number of examples
h_new is the height of the output
w_new is the width of the output
c_new is the number of channels in the output
A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the
output of the previous layer
h_prev is the height of the previous layer
w_prev is the width of the previous layer
c_prev is the number of channels in the previous layer
W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels
for the convolution
kh is the filter height
kw is the filter width
b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied to
the convolution
padding is a string that is either same or valid, indicating the type of
padding used
stride is a tuple of (sh, sw) containing the strides for the convolution
sh is the stride for the height
sw is the stride for the width
you may import numpy as np
Returns: the partial derivatives with respect to the previous layer (dA_prev),
the kernels (dW), and the biases (db), respectively
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ Function Convolution Forward"""
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    sh = stride[0]
    sw = stride[1]
    ph, pw = 0, 0
    if padding == 'same':
        ph = int(((h_new - 1)*sh + kh - h_new)/2) + 1
        pw = int(((w_new - 1)*sw + kw - w_new)/2) + 1
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    for row in range(kh):
        for col in range(kw):
            dW[row, col, :, :] = np.sum(np.multiply(A_prev[:,
                                                           row:row+h_new,
                                                           col:col+w_new,
                                                           :],
                                                    dZ[:, :, :, :]),
                                        axis=(0, 1, 2))
    db = np.sum(A_prev + b, axis=(1, 2, 3))
    dX = np.zeros_like(A_prev)
    dx_1 = np.pad(dX, ((0, 0), (2*ph, 2*ph), (2*pw, 2*pw), (0, 0)),
                  mode='constant', constant_values=0)
    ch = int(np.floor(((h_new - kh + 2*ph) / sh) + 1))
    cw = int(np.floor(((w_new - kw + 2*pw) / sw) + 1))
    m_o = np.arange(0, m)
    for row in range(ch):
        for col in range(cw):
            for n_k in range(c_prev):
                a = row*sh
                b = row*sh + kh
                c = col*sw
                d = col*sw + kw
                dx_1[m_o, row, col, n_k] = np.sum(np.multiply
                                                  (dZ[m_o,
                                                      a:b,
                                                      c:d, ],
                                                   W[:, :, :, n_k]),
                                                  axis=(1, 2, 3))
    return(dx_1, dW, db)
