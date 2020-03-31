#!/usr/bin/env python3
""" FUnction Identity Block
Write a function  that builds an block as described in Deep
Residual Learning for Image Recognition (2015)

A_prev is the output from the previous layer
filters is a tuple or list containing F11, F3, F12, respectively:
F11 is the number of filters in the first 1x1 convolution
F3 is the number of filters in the 3x3 convolution
F12 is the number of filters in the second 1x1 convolution
All convolutions inside the block should be followed by batch
normalization along the channels axis and a rectified linear
activation (ReLU), respectively.
All weights should use he normal initialization
Returns: the activated output of the identity block
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def identity_block(A_prev, filters):
    """Function Inception Block"""
    F11 = filters[0]
    F3 = filters[1]
    F12 = filters[2]
    init = K.initializers.he_normal(seed=None)
    l_0 = K.layers.Conv2D(F11, (1, 1),  padding='same', strides=(1, 1),
                          kernel_initializer=init)(A_prev)
    l_0_b = K.layers.BatchNormalization()(l_0)
    l_0_a = K.layers.Activation('relu')(l_0_b)
    l_1 = K.layers.Conv2D(F3, (3, 3),  padding='same', strides=(1, 1),
                          kernel_initializer=init)(l_0_a)
    l_1_b = K.layers.BatchNormalization()(l_1)
    l_1_a = K.layers.Activation('relu')(l_1_b)
    l_2 = K.layers.Conv2D(F12, (1, 1),  padding='same', strides=(1, 1),
                          kernel_initializer=init)(l_1_a)
    l_2_b = K.layers.BatchNormalization()(l_2)
    out = K.layers.add([l_2_b, A_prev])
    out_a = K.layers.Activation('relu')(out)
    return(out_a)
