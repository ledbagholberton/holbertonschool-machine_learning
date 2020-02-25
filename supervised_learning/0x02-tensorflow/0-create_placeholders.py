#!/usr/bin/env python3
"""FUnction create placeholders """
import tensorflow as tf


def create_placeholders(nx, classes):
    """Function that creates placeholders"""
    x = tf.placeholder("float", [None, nx], name='x')
    y = tf.placeholder("float", [None, classes], name='y')
    return(x, y)
