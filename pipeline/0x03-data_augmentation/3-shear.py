#!/usr/bin/env python3
"""
Write a function that randomly shears an image
image is a 3D tf.Tensor containing the image to shear
intensity is the intensity with which the image should be sheared
Returns the sheared image
"""
import tensorflow as tf
import numpy as np


def shear_image(image, intensity):
    """Rotate image"""
    flip_2 = tf.keras.preprocessing.image.random_shear(image, intensity)
    return flip_2
