#!/usr/bin/env python3
"""
Write a function that :

Z is a tf.tensor containing the input to the generator network
The network should have two layers:
the first layer should have 128 nodes and use relu activation with name layer_1
the second layer should have 784 nodes and use a sigmoid activation with name layer_2
Returns X, a tf.tensor containing the generated image
"""
import tensorflow as tf
import numpy as np


def generator(Z):
    """Function generator creates a simple generator network for MNIST digits"""
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        Nodes_1 = 128
        Nodes_2 = 784
        X = tf.layers.dense(Z, units=Nodes_1, name='layer_1', activation='relu')
        X = tf.layers.dense(X, units=Nodes_2, name='layer_2')
        X = tf.nn.sigmoid(X)
    return(X)
