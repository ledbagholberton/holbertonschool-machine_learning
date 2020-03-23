#!/usr/bin/env python3
""" Function Pool Forward
Write a function  that performs forward propagation over a pooling layer of
a neural network:

A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
the output of the previous layer
m is the number of examples
h_prev is the height of the previous layer
w_prev is the width of the previous layer
c_prev is the number of channels in the previous layer
kernel_shape is a tuple of (kh, kw) containing the size of the kernel for
the pooling
kh is the kernel height
kw is the kernel width
stride is a tuple of (sh, sw) containing the strides for the pooling
sh is the stride for the height
sw is the stride for the width
mode is a string containing either max or avg, indicating whether to perform
maximum or average pooling, respectively
you may import numpy as np
Returns: the output of the pooling layer
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
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
