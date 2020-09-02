#!/usr/bin/env python3
"""
X is a np.ndarray of shape (m, 784) containing the real data input
m is the number of real data samples
epochs is the number of epochs that the each network should be trained for
batch_size is the batch size that should be used during training
Z_dim is the number of dimensions for the randomly generated input
save_path is the path to save the trained generator
Create the tf.placeholder for Z and add it to the graph’s collection
The discriminator and generator training should be altered after one epoch
"""
import tensorflow as tf
import numpy as np
train_generator = __import__('2-train_generator').train_generator
train_discriminator = __import__('3-train_discriminator').train_discriminator
sample_Z = __import__('4-sample_Z').sample_Z


def train_gan(X, epochs, batch_size, Z_dim, save_path='/tmp'): 
    """train a GAN"""
    m = X.shape[0]
    n = X.shape[1]
    Z_place = tf.placeholder(tf.float32, shape=[None, Z_dim])
    for e in range(epochs):
        for b in range(batch_size):
            Z = sample_Z(m, n)
            D_loss, D_solver = train_discriminator(Z, X)
            G_loss, G_solver = train_generator(Z)
            