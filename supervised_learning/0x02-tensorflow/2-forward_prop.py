#!/usr/bin/env python3
"""Function Forward Propagation
This function creates the graph with all the Neuron layers including the
inputs.
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Function that creates the graph for forward propagation
    x are the placeholder with inputs
    layer_sizes is a list with the size of each layer
    activations is a list with the activation function for each layer
    return the prediction - output last layer
    """
    a = create_layer(x, layer_sizes[0], activation=activations[0])
    for i in range(1, len(layer_sizes)):
        a = create_layer(a, layer_sizes[i], activation=activations[i])
    return(a)
