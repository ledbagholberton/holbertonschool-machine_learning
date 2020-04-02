"""Function transition layer
Write a function that builds a transition layer as described
in Densely Connected Convolutional Networks:

X is the output from the previous layer
nb_filters is an integer representing the number of filters in X
compression is the compression factor for the transition layer
Your code should implement compression as used in DenseNet-C
All weights should use he normal initialization
All convolutions should be preceded by Batch Normalization and a
rectified linear activation (ReLU), respectively
Returns: The output of the transition layer and the number of filters
within the output, respectively
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ Function Dense Block"""
    init = K.initializers.he_normal(seed=None)
    l_0_b = K.layers.BatchNormalization()(X)
    l_0_a = K.layers.Activation('relu')(l_0_b)
    l_0 = K.layers.Conv2D(int(nb_filters*compression), (1, 1),  padding='same',
                          kernel_initializer=init)(l_0_a)
    avg1 = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(l_0)
    return(avg1, int(nb_filters*compression))
