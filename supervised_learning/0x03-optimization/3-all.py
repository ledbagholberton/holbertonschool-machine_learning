#!/usr/bin/env python3
"""Several functions required to train a Deep Neural Network"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """Function that creates placeholders"""
    x = tf.placeholder("float", [None, nx], name='x')
    y = tf.placeholder("float", [None, classes], name='y')
    return(x, y)

def create_layer(prev, n, activation):
    """Function that creates a layer"""
    heetal = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    my_layer = tf.layers.Dense(units=n, activation=activation,
                               kernel_initializer=heetal, name='layer')
    return(my_layer(prev))

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

def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of Prediction
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction
    """
    equality = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return(accuracy)

def calculate_loss(y, y_pred):
    """Function that calculates the softmax cross-entropy loss of a prediction
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the loss of the prediction
    """
    return(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred))

def create_train_op(loss, alpha):
    """Function that creates the training operation for the network
    loss is the loss of the network’s prediction
    alpha is the learning rate
    Returns: an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return(train)
