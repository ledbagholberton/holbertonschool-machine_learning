#!/usr/bin/env python3
""" Functions save & load
 saves an entire model:
network is the model to save
filename is the path of the file that the model should be saved to
Returns: None
 loads an entire model:
filename is the path of the file that the model should be loaded from
Returns: the loaded model

"""
import tensorflow.keras as K


def save_model(network, filename):
    """FUnction save_model"""
    network.save(filename)


def load_model(filename):
    """FUnction load_model"""
    return(K.models.load_model(filename))
