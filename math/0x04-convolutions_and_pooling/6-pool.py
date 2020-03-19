#!/usr/bin/env python3
"""Function convolve pooling
Write a function that performs pooling on images:

images is a numpy.ndarray with shape (m, h, w, c) containing multiple images
m is the number of images
h is the height in pixels of the images
w is the width in pixels of the images
c is the number of channels in the image
kernel_shape is a tuple of (kh, kw) containing the kernel shape for the pooling
kh is the height of the kernel
kw is the width of the kernel
stride is a tuple of (sh, sw)
sh is the stride for the height of the image
sw is the stride for the width of the image
mode indicates the type of pooling
max indicates max pooling
avg indicates average pooling
You are only allowed to use two for loops; any other loops of any kind are not
allowed
Returns: a numpy.ndarray containing the pooled images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Function convolution pooling"""
    m, h, w, c = images.shape
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
                new_conv[m_only, row, col] = np.max(images[m_only,
                                                           a:b,
                                                           c:d,
                                                           ], axis=(1, 2))
            else:
                new_conv[m_only, row, col] = np.average(images[m_only,
                                                               a:b,
                                                               c:d,
                                                               ], axis=(1, 2))
    return(new_conv)
