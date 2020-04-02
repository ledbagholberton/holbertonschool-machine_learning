#!/usr/bin/env python3
"""Function dense block
Write a function  that builds a dense block as described in Densely
Connected Convolutional Networks:

X is the output from the previous layer
nb_filters is an integer representing the number of filters in X
growth_rate is the growth rate for the dense block
layers is the number of layers in the dense block
You should use the bottleneck layers used for DenseNet-B
All weights should use he normal initialization
All convolutions should be preceded by Batch Normalization and a rectified
linear activation (ReLU), respectively
Returns: The concatenated output of each layer within the Dense Block and the
number of filters within the concatenated outputs, respectively
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ Function Dense Block"""
    init = K.initializers.he_normal(seed=None)
    for my_layer in range(layers):
        l_0_b = K.layers.BatchNormalization()(X)
        l_0_a = K.layers.Activation('relu')(l_0_b)
        l_0 = K.layers.Conv2D(growth_rate * 4, (1, 1),  padding='same',
                              kernel_initializer=init)(l_0_a)
        l_1_b = K.layers.BatchNormalization()(l_0)
        l_1_a = K.layers.Activation('relu')(l_1_b)
        l_1 = K.layers.Conv2D(growth_rate, (3, 3),  padding='same',
                              kernel_initializer=init)(l_1_a)
        X = K.layers.concatenate([X, l_1])
        nb_filters = nb_filters + growth_rate
    return(X, nb_filters)
