#!/usr/bin/env python3
"""
Write a function  that performs a random crop of an image:

image is a 3D tf.Tensor containing the image to crop
size is a tuple containing the size of the crop
Returns the cropped image
"""
import tensorflow as tf
import numpy as np


def crop_image(image, size):
    """Crop image"""
    height = size[0]
    width = size[1]
    offset_h = np.random.randint(0, height)
    offset_w = np.random.randint(0, width)
    flip_2 = tf.image.crop_to_bounding_box(image, offset_h,
                                           offset_w, height,
                                           width)
    return flip_2
