#!/usr/bin/env python3
"""
X is a tf.tensor containing the input to the discriminator network
The network should have two layers:
the first layer should have 128 nodes and use 
relu activation with name layer_1
the second layer should have 1 node and use a 
sigmoid activation with name layer_2
All variables in the network should have the
scope discriminator with reuse=tf.AUTO_REUSE
Returns Y, a tf.tensor containing the classification made by the discriminator
"""
import tensorflow as tf
import numpy as np


def discriminator(X):
    """Function discriminator"""
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        Nodes_1 = 128
        Nodes_2 = 1
        Y = tf.layers.dense(Y, units=Nodes_1, name='layer_1', activation='relu')
        Y = tf.layers.dense(Y, units=Nodes_2, name='layer_2')
        Y = tf.nn.sigmoid(Y)
    return(Y)
