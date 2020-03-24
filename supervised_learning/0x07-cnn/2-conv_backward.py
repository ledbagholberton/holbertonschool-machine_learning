#!/usr/bin/env python3
""" Function Pool Forward
Write a function 
that performs back propagation over a convolutional layer of a neural network:

dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial
derivatives with respect to the unactivated output of the convolutional layer
m is the number of examples
h_new is the height of the output
w_new is the width of the output
c_new is the number of channels in the output
A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the
output of the previous layer
h_prev is the height of the previous layer
w_prev is the width of the previous layer
c_prev is the number of channels in the previous layer
W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels for
the convolution
kh is the filter height
kw is the filter width
b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied to
the convolution
padding is a string that is either same or valid, indicating the type of padding used
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
    m, h, w, c = A_prev.shape
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]
    ch = int(np.floor(((h - kh) / sh) + 1))
    cw = int(np.floor(((w - kw) / sw) + 1))
    new_conv = np.zeros((m, ch, cw, c))
    m_only = np.arange(0, m)
    for row in range(ch):
        for col in range(cw):
            a = row*sh
            b = row*sh + kh
            c = col*sw
            d = col*sw + kw
            if mode == 'max':
                new_conv[m_only, row, col] = np.max(A_prev[m_only,
                                                           a:b,
                                                           c:d,
                                                           ], axis=(1, 2))
            else:
                new_conv[m_only, row, col] = np.average(A_prev[m_only,
                                                               a:b,
                                                               c:d,
                                                               ], axis=(1, 2))
    return(new_conv)
