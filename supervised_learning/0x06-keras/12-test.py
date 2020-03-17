#!/usr/bin/env python3
""" Functions test neural network
network is the network model to test
data is the input data to test the model with
labels are the correct one-hot labels of data
verbose is a boolean that determines if output should be printed
during the testing process
Returns: the loss and accuracy of the model with the testing data
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """FUnction test neural network"""
    scores = network.evaluate(data, labels, verbose=verbose)
    return(scores)
