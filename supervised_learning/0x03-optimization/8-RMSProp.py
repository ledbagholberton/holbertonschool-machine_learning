#!/usr/bin/env python3
""" Function RMSProp
Creates the training operation for a neural network in tensorflow
using the RMSProp optimization algorithm:
loss is the loss of the network
alpha is the learning rate
beta2 is the RMSProp weight
epsilon is a small number to avoid division by zero
Returns: the RMSProp optimization operation
"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Function RMSProp"""
    train = tf.train.RMSPropOptimizer(alpha, beta2, epsilon).minimize(loss)
    return(train)
