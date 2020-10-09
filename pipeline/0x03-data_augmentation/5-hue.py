#!/usr/bin/env python3
"""
Write a function  that changes the hue of an image:

image is a 3D tf.Tensor containing the image to change
delta is the amount the hue should change
Returns the altered image
"""
import tensorflow as tf
import numpy as np


def change_hue(image, delta):
    """Rotate image"""
    flip_2 = tf.image.random_hue(image, -delta)
    return flip_2
