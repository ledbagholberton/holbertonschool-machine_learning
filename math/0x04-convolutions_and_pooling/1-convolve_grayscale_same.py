#!/usr/bin/env python3
"""Function convolve grayscale same
Write a function def convolve_grayscale_same(images, kernel): that performs
a same
convolution on grayscale images:

images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale
images
m is the number of images
h is the height in pixels of the images
w is the width in pixels of the images
kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the
convolution
kh is the height of the kernel
kw is the width of the kernel
if necessary, the image should be padded with 0’s
You are only allowed to use two for loops; any other loops of any kind are
not allowed
Returns: a numpy.ndarray containing the convolved images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function convolution same"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    p = int((kh-1)/2)
    ch = h + 2*p - kh + 1
    cw = w + 2*p - kw + 1
    new_images = np.pad(images, ((0, 0), (p, p), (p, p)), mode='constant',
                        constant_values=0)
    m, h, w = new_images.shape
    new_conv = np.zeros((m, ch, cw))
    m_only = np.arange(0, m)
    for row in range(ch):
        for col in range(cw):
            new_conv[m_only, row, col] = np.sum(np.multiply
                                                (new_images[m_only,
                                                            row:row + kh,
                                                            col:col + kw],
                                                 kernel), axis=(1, 2))
    return(new_conv)
