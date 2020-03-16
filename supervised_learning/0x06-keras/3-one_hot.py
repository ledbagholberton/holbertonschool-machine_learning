#!/usr/bin/env python3
""" Function one_hot
Write a function that converts a label
vector into a one-hot matrix:
The last dimension of the one-hot matrix must be the number of classes
Returns: the one-hot matrix
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ Function one-hot"""
    one_hot = K.utils.to_categorical(labels, classes)
    return (one_hot)
