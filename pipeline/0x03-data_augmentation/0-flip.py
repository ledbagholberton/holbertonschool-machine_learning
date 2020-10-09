#!/usr/bin/env python3
"""
Flips an image horizontally:

image is a 3D tf.Tensor containing the image to flip
Returns the flipped image
"""
import tensorflow as tf
import numpy as np


def flip_image(image):
    """Flip image
    """
    flip_2 = tf.image.flip_left_right(image)
    return flip_2
