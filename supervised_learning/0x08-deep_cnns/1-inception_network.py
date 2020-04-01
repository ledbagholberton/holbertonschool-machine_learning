#!/usr/bin/env python3
""" FUnction Inception Block
Write a function : that builds the inception network
as described in Going Deeper with Convolutions (2014):

You can assume the input data will have shape (224, 224, 3)
All convolutions inside and outside the inception block should use a rectified
linear activation (ReLU)
You may use inception_block = __import__('0-inception_block').inception_block
Returns: the keras model
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Function Inception Block"""
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)
    conv1 = K.layers.Conv2D(64, (7, 7),  padding='same', strides=(2, 2),
                            kernel_initializer=init, activation='relu')(X)
    mp1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same')(conv1)
    conv2 = K.layers.Conv2D(64, (1, 1),  padding='same',
                            kernel_initializer=init, activation='relu')(mp1)
    conv2_1 = K.layers.Conv2D(192, (3, 3),  padding='same',
                              kernel_initializer=init, activation='relu')(conv2)
    mp2 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same')(conv2_1)
    inc3a = inception_block(mp2, [64, 96, 128, 16, 32, 32])
    inc3b = inception_block(inc3a, [128, 128, 192, 32, 96, 64])
    mp3 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same')(inc3b)
    inc4a = inception_block(mp3, [192, 96, 208, 16, 48, 64])
    inc4b = inception_block(inc4a, [160, 112, 224, 24, 64, 64])
    inc4c = inception_block(inc4b, [128, 128, 256, 24, 64, 64])
    inc4d = inception_block(inc4c, [112, 144, 288, 32, 64, 64])
    inc4e = inception_block(inc4d, [256, 160, 320, 32, 128, 128])
    mp4 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same')(inc4e)
    inc5a = inception_block(mp4, [256, 160, 320, 32, 128, 128])
    inc5b = inception_block(inc5a, [384, 192, 384, 48, 128, 128])
    avg1 = K.layers.AveragePooling2D((7, 7), strides=None)(inc5b)
    drop1 = K.layers.Dropout(0.4)(avg1)
    dense_1 = K.layers.Dense(1000, activation='softmax',
                             kernel_initializer=init)(drop1)
    model = K.models.Model(inputs=X, outputs=dense_1)
    return(model)
