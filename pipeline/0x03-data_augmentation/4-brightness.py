#!/usr/bin/env python3
"""
Write a function  that randomly changes the brightness of an image:

image is a 3D tf.Tensor containing the image to change
max_delta is the maximum amount the image should be brightened (or darkened)
Returns the altered image
"""
import tensorflow as tf
import numpy as np


def change_brightness(image, max_delta):
    """Rotate image"""
    flip_2 = tf.image.random_brightness(image, max_delta)
    return flip_2
