#!/usr/bin/env python3
""" Function dropout_create_layer
creates a layer of a neural network using dropout:

prev is a tensor containing the output of the previous layer
n is the number of nodes the new layer should contain
activation is the activation function that should be used on the layer
keep_prob is the probability that a node will be kept
Returns: the output of the new layer
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """FUnction dropout_create_layer"""
    heetal = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    drop_regularizer = tf.layers.Dropout(keep_prob)
    my_layer = tf.layers.Dense(units=n, activation=activation,
                               kernel_initializer=heetal, name='layer',
                               kernel_regularizer=drop_regularizer)
    return(my_layer(prev))
