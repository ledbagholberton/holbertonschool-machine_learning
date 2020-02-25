#!/usr/bin/env python3
"""Function Accuracy
This function calculate the accuracy of a prediction
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of Prediction
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction
    """
    equality = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return(accuracy)
