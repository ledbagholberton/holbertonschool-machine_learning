#!/usr/bin/env python3
"""FUnction create a layer """
import tensorflow as tf


def create_layer(prev, n, activation):
    """Function that creates a layer"""
    heetal = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    my_layer = tf.layers.Dense(units=n, activation=activation,
                               kernel_initializer=heetal, name='layer')
    return(my_layer(prev))
