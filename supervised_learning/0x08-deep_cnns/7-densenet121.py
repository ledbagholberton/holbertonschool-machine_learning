#!/usr/bin/env python3
"""
Write a function  that builds the DenseNet-121 architecture as described
in Densely Connected Convolutional Networks:

growth_rate is the growth rate
compression is the compression factor
You can assume the input data will have shape (224, 224, 3)
All convolutions should be preceded by Batch Normalization and a rectified
linear activation (ReLU), respectively
All weights should use he normal initialization
Returns: the keras model
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Function Densenet121"""
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)
    list_ly = [6, 12, 24, 16]
    cv1_b = K.layers.BatchNormalization()(X)
    cv1_a = K.layers.Activation('relu')(cv1_b)
    conv1 = K.layers.Conv2D(64, (7, 7),  padding='same', strides=(2, 2),
                            kernel_initializer=init)(cv1_a)
    mp1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same')(conv1)
    Y1, nb_filters1 = dense_block(mp1, 64, growth_rate, list_ly[0])
    Y2, nb_filters2 = transition_layer(Y1, nb_filters1, compression)
    Y3, nb_filters3 = dense_block(Y2, nb_filters2, growth_rate, list_ly[1])
    Y4, nb_filters4 = transition_layer(Y3, nb_filters3, compression)
    Y5, nb_filters5 = dense_block(Y4, nb_filters4, growth_rate, list_ly[2])
    Y6, nb_filters6 = transition_layer(Y5, nb_filters5, compression)
    Y7, nb_filters7 = dense_block(Y6, nb_filters6, growth_rate, list_ly[3])
    avg1 = K.layers.AveragePooling2D((7, 7), strides=None, padding='same')(Y7)
    dense_1 = K.layers.Dense(1000, activation='softmax',
                             kernel_initializer=init)(avg1)
    model = K.models.Model(inputs=X, outputs=dense_1)
    return(model)
