#!/usr/bin/env python3
""" Functions save & load weights
Saves a model’s weights:
network is the model whose weights should be saved
filename is the path of the file that the weights should be saved to
save_format is the format in which the weights should be saved
Returns: None
Loads a model’s weights:
network is the model to which the weights should be loaded
filename is the path of the file that the weights should be loaded from
Returns: None
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """FUnction save_weights"""
    fullname = filename + '.' + save_format
    network.save_weights(fullname)


def load_weights(network, filename):
    """FUnction load_weights"""
    network.load_weights(filename)
