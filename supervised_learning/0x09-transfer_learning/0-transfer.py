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
import h5py


if __name__ == '__main__':
    (X, Y), _ = K.datasets.cifar10.load_data()
    X_train = K.applications.densenet.preprocess_input(X)
    y_train = K.utils.to_categorical(Y, 10)
    model = K.applications.densenet.DenseNet121(
        weights='imagenet', 
        include_top=False,
        input_shape=(32,32,3),
        classes=1000
    )
    for layer in model.layers:
        layer.trainable = False
    out_0 = model.layers[-1].output
    out_1 = K.layers.Flatten()(out_0)
    out_2 = K.layers.Activation('relu')(out_1)
    drop2 = K.layers.Dropout(rate=0.5)(out_2)
    dense = K.layers.Dense(10, activation='softmax')(drop2)
    model = K.models.Model(inputs=model.inputs, outputs=dense)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    """checkpoint = K.callbacks.ModelCheckpoint(
        'model.h5', 
        monitor='val_acc', 
        verbose=0, 
        save_best_only=True, 
        save_weights_only=False,
        mode='auto'
    )"""
    model.summary()
    # Train the model
    hist = model.fit(
        x=X_train,
        y=y_train,
        batch_size=1,
        epochs=1,
        verbose=0
    )
    model.save('cifar10.h5')

def preprocess_data(X, Y):
    """Function preprocess data """
    import tensorflow.keras as K

    x_p = K.applications.densenet.preprocess_input(X)
    y_p = K.utils.to_categorical(Y)
    return (x_p, y_p)
