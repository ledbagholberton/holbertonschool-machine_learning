#!/usr/bin/env python3
""" FUnction Inception Block
Write a function  that builds the ResNet-50 architecture as described
in Deep Residual Learning for Image Recognition (2015):

You can assume the input data will have shape (224, 224, 3)
All convolutions inside and outside the blocks should be followed by
batch normalization along the channels axis and a rectified linear activation
(ReLU), respectively.
All weights should use he normal initialization
Returns: the keras model
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Function Inception Block"""
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)
    conv1 = K.layers.Conv2D(64, (7, 7),  padding='same', strides=(2, 2),
                            kernel_initializer=init)(X)
    conv1_b = K.layers.BatchNormalization()(conv1)
    conv1_a = K.layers.Activation('relu')(conv1_b)
    mp1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same')(conv1_a)
    conv2_1 = projection_block(mp1, [64, 64, 256], 1)
    conv2_2 = identity_block(conv2_1, [64, 64, 256])
    conv2_3 = identity_block(conv2_2, [64, 64, 256])

    conv3_1 = projection_block(conv2_3, [128, 128, 512])
    conv3_2 = identity_block(conv3_1, [128, 128, 512])
    conv3_3 = identity_block(conv3_2, [128, 128, 512])
    conv3_4 = identity_block(conv3_3, [128, 128, 512])

    conv4_1 = projection_block(conv3_4, [256, 256, 1024])
    conv4_2 = identity_block(conv4_1, [256, 256, 1024])
    conv4_3 = identity_block(conv4_2, [256, 256, 1024])
    conv4_4 = identity_block(conv4_3, [256, 256, 1024])
    conv4_5 = identity_block(conv4_4, [256, 256, 1024])
    conv4_6 = identity_block(conv4_5, [256, 256, 1024])

    conv5_1 = projection_block(conv4_6, [512, 512, 2048])
    conv5_2 = identity_block(conv5_1, [512, 512, 2048])
    conv5_3 = identity_block(conv5_2, [512, 512, 2048])

    avg1 = K.layers.AveragePooling2D((7, 7), strides=(1, 1))(conv5_3)
    dense_1 = K.layers.Dense(1000, activation='softmax',
                             kernel_initializer=init)(avg1)
    model = K.models.Model(inputs=X, outputs=dense_1)
    return(model)
