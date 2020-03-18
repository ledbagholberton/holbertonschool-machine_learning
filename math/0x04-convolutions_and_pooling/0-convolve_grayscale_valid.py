#!/usr/bin/env python3
""" Function convolve_grayscale
Write a function that performs a valid convolution on grayscale images:

images is a numpy.ndarray with shape (m, h, w) containing multiple
grayscale images
m is the number of images
h is the height in pixels of the images
w is the width in pixels of the images
kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
for the convolution
kh is the height of the kernel
kw is the width of the kernel
You are only allowed to use two for loops; any other loops of any kind
are not allowed
Returns: a numpy.ndarray containing the convolved images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Function convolution"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ch = h - kh + 1
    cw = w - kw + 1
    new_conv = np.zeros((m, ch, cw))
    m_only = np.arange(0, m)
    #m_only = images[0]
    print(m_only)
    for row in range(cw):
        for col in range(ch):
            new_conv[m_only, row, col] = np.sum(np.multiply(images[m_only, row:row + kh, col:col + kw], kernel), axis=(1,2))
    return(new_conv)
