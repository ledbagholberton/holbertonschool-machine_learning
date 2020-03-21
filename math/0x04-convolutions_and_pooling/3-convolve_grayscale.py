#!/usr/bin/env python3
"""Function convolve grayscale strade
Write a function  that performs a convolution on grayscale images

images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale
images
m is the number of images
h is the height in pixels of the images
w is the width in pixels of the images
kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the
convolution
kh is the height of the kernel
kw is the width of the kernel
padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
if ‘same’, performs a same convolution
if ‘valid’, performs a valid convolution
if a tuple:
ph is the padding for the height of the image
pw is the padding for the width of the image
the image should be padded with 0’s
stride is a tuple of (sh, sw)
sh is the stride for the height of the image
sw is the stride for the width of the image
You are only allowed to use two for loops; any other loops of any kind are
not allowed Hint: loop over i and j
Returns: a numpy.ndarray containing the convolved images
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Function convolution with padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh = stride[0]
    sw = stride[1]
    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]
    elif padding == 'same':
        if kh % 2 != 0:
            ph = int([(h - 1)*sh + kh - h]/2)
        else:
            ph = int((h*sh + kh - h)/2)
        if kw % 2 != 0:
            pw = int([(w - 1)*sw + kw - w]/2)
        else:
            pw = int((w*sw + kw - w)/2)
    else:
        ph, pw = 0, 0
    if kh % 2 != 0:
        ch = int(np.floor(((h - kh + 2*ph) / sh) + 1))
    else:
        ch = int(np.floor(((h - kh + 2*ph) / sh)))
    if kw % 2 != 0:
        cw = int(np.floor(((w - kw + 2*pw) / sw) + 1))
    else:
        cw = int(np.floor(((w - kw + 2*pw) / sw)))
    new_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant',
                        constant_values=0)
    m, new_h, new_w = new_images.shape
    new_conv = np.zeros((m, ch, cw))
    m_only = np.arange(0, m)
    for row in range(ch):
        for col in range(cw):
            a = row * sh
            b = row * sh + kh
            c = col * sw
            d = col * sw + kw
            new_conv[m_only, row, col] = np.sum(np.multiply
                                                (new_images[m_only,
                                                            a: b,
                                                            c: d],
                                                 kernel), axis=(1, 2))
    return(new_conv)
