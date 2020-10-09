#!/usr/bin/env python3
"""
Write a function  that rotates an image by 90
degrees counter-clockwise:

image is a 3D tf.Tensor containing the image to rotate
Returns the rotated image
"""
import tensorflow as tf
import numpy as np


def rotate_image(image):
    """Rotate image"""
    flip_2 = tf.image.rot90(image, k=1, name=None)
    return flip_2
