#!/usr/bin/env python3
""" Function l2_reg_cost
calculates the cost of a neural network with L2 regularization:
cost is a tensor containing the cost of the network without
L2 regularization
Returns: a tensor containing the cost of the network accounting for
L2 regularization
"""
import numpy as np
import tensorflow as tf


def l2_reg_cost(cost):
    """Function l2_reg_cost"""
    return(cost + tf.losses.get_regularization.loss())
