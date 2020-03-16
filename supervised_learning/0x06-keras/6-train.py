#!/usr/bin/env python3
""" Function train_model
Function that trains a model using mini-batch gradient descent

network is the model to train
data is a numpy.ndarray of shape (m, nx) containing the input data
labels is a one-hot numpy.ndarray of shape (m, classes) containing
the labels of data
batch_size is the size of the batch used for mini-batch gradient descent
epochs is the number of passes through data for mini-batch gradient descent
verbose is boolean determines if output should be printed during training
shuffle is boolean determines whether to shuffle the batches every epoch.
validation_data is the data to validate the model with, if not None
update the function
early_stopping is a boolean that indicates whether early stopping exist
early stopping should only be performed if validation_data exists
early stopping should be based on validation loss
patience is the patience used for early stopping

Returns: the History object generated after training the model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None,
                early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ Function train_model"""
    callbacks = [K.callbacks.EarlyStopping(patience=patience,
                                           monitor='val_loss')]
    trained_network = network.fit(data, labels, nb_epoch=epochs,
                                  batch_size=batch_size, shuffle=shuffle,
                                  verbose=verbose,
                                  validation_data=validation_data)
    return (trained_network)
