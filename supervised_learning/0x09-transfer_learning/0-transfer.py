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
    y_train = K.utils.to_categorical(Y, 10)
    expand_x_train = np.fliplr(X_train)
    X_train = np.concatenate([X_train, expand_x_train])
    y_train = np.concatenate([y_train, y_train])
    model = K.applications.vgg16.VGG16(
        weights='imagenet', 
        include_top=False,
        input_shape=(32,32,3),
        classes=10
    )
    """for layer in model.layers[:300]:
        layer.trainable = False
    for layer in model.layers[300:]:
        layer.trainable = True"""
    out_0 = model.layers[-1].output
    drop_1 = K.layers.Dropout(0.4)(out_0)
    out_1 = K.layers.Flatten()(drop_1)
    out_2 = K.layers.Activation('relu')(out_1)
    dense_1 = K.layers.Dense(256, activation='relu')(out_2)
    drop_1 = K.layers.Dropout(0.4)(dense_1)
    dense_3 = K.layers.Dense(10, activation='softmax')(drop_1)
    model = K.models.Model(inputs=model.inputs, outputs=dense_3)
    count = 0
    for layer in model.layers[:15]:
        layer.trainable = False
    for layer in model.layers[16:]:
        layer.trainable = True
    model.summary()
    print("Numero de capas", count)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=K.optimizers.RMSprop(lr=1e-4),
        metrics=['accuracy']
    )
    lrr=K.callbacks.ReduceLROnPlateau(
                monitor='acc',
                factor=.01,
                patience=2,
                min_lr=1e-5)
    nb = 1000
    epochs = 5
    hist = model.fit(
        x=X_train,
        y=y_train,
        batch_size=nb,
        epochs=epochs,
        verbose=1,
        callbacks=[lrr]
    )
    model.save('cifar10.h5')

def preprocess_data(X, Y):
    """Function preprocess data """
    import tensorflow.keras as K

    x_p = K.applications.densenet.preprocess_input(X)
    y_p = K.utils.to_categorical(Y, 10)
    return (x_p, y_p)
