#!/usr/bin/env python3
""" Function Projection Block
Write a function  that builds a projection block as described in Deep Residual
Learning for Image Recognition (2015):
A_prev is the output from the previous layer
filters is a tuple or list containing F11, F3, F12, respectively:
F11 is the number of filters in the first 1x1 convolution
F3 is the number of filters in the 3x3 convolution
F12 is the number of filters in the second 1x1 convolution as well as the 1x1
convolution in the shortcut connection
s is the stride of the first convolution in both the main path and the shortcut
connection
All convolutions inside the block should be followed by batch normalization}
along the channels axis and a rectified linear activation (ReLU), respectively.
All weights should use he normal initialization
Returns: the activated output of the projection block
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ Function Porjetin Block"""
    F11 = filters[0]
    F3 = filters[1]
    F12 = filters[2]
    init = K.initializers.he_normal(seed=None)
    l_0 = K.layers.Conv2D(F11, (1, 1),  padding='same', strides=(s, s),
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
    alt_0 = K.layers.Conv2D(F12, (1, 1),  padding='same', strides=(s, s),
                            kernel_initializer=init)(A_prev)
    alt_0_b = K.layers.BatchNormalization()(alt_0)
    out = K.layers.add([l_2_b, alt_0_b])
    out_a = K.layers.Activation('relu')(out)
    return(out_a)
