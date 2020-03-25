#!/usr/bin/env python3
"""
Write a function def lenet5(X): that builds a modified version of
the LeNet-5 architecture using keras:

X is a K.Input of shape (m, 28, 28, 1) containing the input images
for the network
m is the number of images
The model should consist of the following layers in order:
Convolutional layer with 6 kernels of shape 5x5 with same padding
Max pooling layer with kernels of shape 2x2 with 2x2 strides
Convolutional layer with 16 kernels of shape 5x5 with valid padding
Max pooling layer with kernels of shape 2x2 with 2x2 strides
Fully connected layer with 120 nodes
Fully connected layer with 84 nodes
Fully connected softmax output layer with 10 nodes
All layers requiring initialization should initialize their kernels
with the he_normal initialization method
All hidden layers requiring activation should use the relu activation}
function.
You may import tensorflow.keras as K
Returns: a K.Model compiled to use Adam optimization (with default
hyperparameters) and accuracy metrics
"""
import tensorflow.keras as K


def lenet5(x):
    """Function lenet5"""
    init = K.initializers.he_normal(seed=None)
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5),
                            padding='same', activation='relu',
                            kernel_initializer=init)(x)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5),
                            padding='valid', activation='relu',
                            kernel_initializer=init)(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    flatt = K.layers.Flatten()(pool2)
    fc1 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer=init)(flatt)
    fc2 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer=init)(fc1)
    fc3 = K.layers.Dense(units=10, activation='softmax',
                         kernel_initializer=init)(fc2)
    model = K.models.Model(x, fc3)
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])
    return(model)
