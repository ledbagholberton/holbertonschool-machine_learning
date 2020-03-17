#!/usr/bin/env python3
""" Function build_model
Builds a neural network with the Keras library

nx is the number of input features to the network
layers is a list containing the number of nodes in each layer of the network
activations is a list containing the activation functions used for each layer
of the network
lambtha is the L2 regularization parameter
keep_prob is the probability that a node will be kept for dropout
You are not allowed to use the Sequential class
Returns: the keras model
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Function build_model """
    inputs = K.Input(shape=(nx,))
    k_reg = K.regularizers.l2(lambtha)
    model = K.layers.Dense(layers[0], activation=activations[0],
                           kernel_regularizer=k_reg)(inputs)
    for i in range(1, len(layers)):
        model = K.layers.Dropout(rate=1-keep_prob)(model)
        model = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=k_reg)(model)
    return (K.Model(inputs=inputs, outputs=model))
