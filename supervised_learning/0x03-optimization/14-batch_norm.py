#!/usr/bin/env python3
"""  Function Batch Normalization
Creates a batch normalization layer for a neural network in tensorflow:
prev is the activated output of the previous layer
n is the number of nodes in the layer to be created
activation is the activation function that should be used on the output
of the layer
Returns: a tensor of the activated output for the layer
"""


import tensorflow as tf
import numpy as np


def create_batch_norm_layer(prev, n, activation):
    """Function Batch Normalization"""
    my_layer = tf.layers.Dense(units=n, activation=activation)
    mean, variance = tf.nn.moments(my_layer(prev), axes=[0])
    BN2 = tf.nn.batch_normalization(my_layer(prev), mean=mean,
                                    variance=variance,
                                    variance_epsilon=0.00001,
                                    offset=None, scale=None)
    return(BN2)
