#!/usr/bin/env python3
""" Function Pool Backward
Write a function that performs back propagation over a pooling layer:

dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial
derivatives with respect to the output of the pooling layer
m is the number of examples
h_new is the height of the output
w_new is the width of the output
c is the number of channels
A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing the output
of the previous layer
h_prev is the height of the previous layer
w_prev is the width of the previous layer
kernel_shape is a tuple of (kh, kw) containing the size of the kernel for the
pooling
kh is the kernel height
kw is the kernel width
stride is a tuple of (sh, sw) containing the strides for the pooling
sh is the stride for the height
sw is the stride for the width
mode is a string containing either max or avg, indicating whether to perform
maximum
or average pooling, respectively
you may import numpy as np
Returns: the partial derivatives with respect to the previous layer (dA_prev)
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """FUnction pool backward"""
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    x_patches = A_prev.reshape(A_prev.shape[0], A_prev.shape[1]//2,
                               kh, A_prev.shape[2]//2, kw, A_prev.shape[3])
    if mode is 'max':
        out = x_patches.max(axis=1).max(axis=2)
        mask = np.isclose(A_prev, np.repeat(np.repeat(out, kh, axis=1),
                          kw, axis=2)).astype(int)
        return mask*(np.repeat(np.repeat(dA, kh, axis=1), kw, axis=2))
    out = x_patches.mean(axis=1).mean(axis=2)
    mask = np.ones_like(A_prev)*(1 / (kh * kw))
    return mask*(np.repeat(np.repeat(dA, kh, axis=1), kw, axis=2))
