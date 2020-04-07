#!/usr/bin/env python3
"""FUnction transfer
Write a python script that trains a convolutional neural network to
classify the CIFAR 10 dataset:

You must use one of the applications listed in Keras Applications
Your script must save your trained model in the current working directory
as cifar10.h5
Your saved model should be compiled
Your saved model should have a validation accuracy of 88% or higher
Your script should not run when the file is imported
Hint: The training may take a while, start early!
In the same file, write a function def preprocess_data(X, Y): that
pre-processes the data for your model:

X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
where m is the number of data points
Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
Returns: X_p, Y_p
X_p is a numpy.ndarray containing the preprocessed X
Y_p is a numpy.ndarray containing the preprocessed Y
"""
import tensorflow.keras as K
import numpy as np


if __name__ == '__main__':
    (X, Y), _ = K.datasets.cifar10.load_data()
    X_train = K.applications.densenet.preprocess_input(X)
    y_train = K.utils.to_categorical(Y)
    model = K.applications.densenet.DenseNet121(
        weights='imagenet', 
        include_top=False,
        input_shape=(32,32,3),
        classes=10
    )
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.summary()
    """ checkpoint = callbacks.ModelCheckpoint(
        'model.h5', 
        monitor='val_acc', 
        verbose=0, 
        save_best_only=True, 
        save_weights_only=False,
        mode='auto'
    )"""
    # Train the model
    hist = model.fit(
        x=X_train,
        y=y_train,
        validation_split=0.1,
        batch_size=64,
        epochs=50,
        callbacks=None,
        verbose=1
    )
    hist.save_weights('cifar10', save_format='h5')

def preprocess_data(X, Y):
    """Function preprocess data """
    import tensorflow.keras as K

    return(K.applications.densenet.preprocess_input(X), K.utils.to_categorical(Y))
