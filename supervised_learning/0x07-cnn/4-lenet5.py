#!/usr/bin/python3
"""
Write a function  that builds a modified version of the
LeNet-5 architecture using tensorflow:

x is a tf.placeholder of shape (m, 28, 28, 1) containing the input images
for the network
m is the number of images
y is a tf.placeholder of shape (m, 10) containing the one-hot labels for
the network
The model should consist of the following layers in order:
Convolutional layer with 6 kernels of shape 5x5 with same padding
Max pooling layer with kernels of shape 2x2 with 2x2 strides
Convolutional layer with 16 kernels of shape 5x5 with valid padding
Max pooling layer with kernels of shape 2x2 with 2x2 strides
Fully connected layer with 120 nodes
Fully connected layer with 84 nodes
Fully connected softmax output layer with 10 nodes
All layers requiring initialization should initialize their kernels with the
he_normal initialization method: 
tf.contrib.layers.variance_scaling_initializer()
All hidden layers requiring activation should use the relu activation function
you may import tensorflow as tf
you may NOT use tf.keras
Returns:
a tensor for the softmax activated output
a training operation that utilizes Adam optimization
(with default hyperparameters)
a tensor for the loss of the netowrk
a tensor for the accuracy of the network
"""
import numpy as np
import tensorflow as tf


def lenet5(x, y):
    """Function lenet5"""
    heetal = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    conv1 = tf.layers.conv2(x, 6, 5, padding='same', activation='Relu', kernel_initializer=heetal)
    pool1 = tf.layers.MaxPooling2D((2,2), (2,2))(conv1)
    conv2 = tf.layers.conv2(pool1, 16, 5, padding='valid', activation='Relu', kernel_initializer=heetal)
    pool2 = tf.layers.MaxPooling2D((2,2), (2,2))(conv2)
    fc1 = tf.layers.Dense(units=120, activation='relu', kernel_initializer=heetal)(pool2)
    fc2 = tf.layers.Dense(units=84, activation='relu', kernel_initializer=heetal)(fc1)
    fc3 = tf.layers.Dense(units=10, activation='softmax', kernel_initializer=heetal)(fc2)
    equality = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=fc3)
    train = tf.train.AdamOptimizer().minimize(loss)
    return(fc3, train, loss, accuracy)
