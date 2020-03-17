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
learning_rate_decay is a boolean that indicates whether learning rate decay
should be used
learning rate decay should only be performed if validation_data exists
the decay should be performed using inverse time decay
the learning rate should decay in a stepwise fashion after each epoch
each time the learning rate updates, Keras should print a message
alpha is the initial learning rate
decay_rate is the decay rate

Returns: the History object generated after training the model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """ Function train_model"""
    callbacks = []
    if early_stopping is True and validation_data is not None:
        early_cb = K.callbacks.EarlyStopping(patience=patience,
                                             monitor='val_loss')
        callbacks.append(early_cb)
    if learning_rate_decay is True and validation_data is not None:
        lr_cb = K.callbacks.LearningRateScheduler(schedule(alpha, decay_rate,
                                                  epochs), 1)
        callbacks.append(lr_cb)
    trained_network = network.fit(data, labels, nb_epoch=epochs,
                                  batch_size=batch_size, shuffle=shuffle,
                                  verbose=verbose,
                                  validation_data=validation_data,
                                  callbacks=callbacks)
    return (trained_network)


def schedule(alpha, decay_rate, epochs):
    """Function support callback Learning Rate"""
    return (alpha / (1 + decay_rate * epochs))
