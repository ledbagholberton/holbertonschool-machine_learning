#!/usr/bin/env python3
"""
Write a function def train_gan(X, epochs, batch_size, Z_dim): that trains a GAN:

X is a np.ndarray of shape (m, 784) containing the real data input
m is the number of real data samples
epochs is the number of epochs that the each network should be trained for
batch_size is the batch size that should be used during training
Z_dim is the number of dimensions for the randomly generated input
save_path is the path to save the trained generator
Create the tf.placeholder for Z and add it to the graph’s collection
The discriminator and generator training should be altered after one epoch
You can use the following imports:
train_generator = __import__('2-train_generator').train_generator
train_discriminator = __import__('3-train_discriminator').train_discriminator
sample_Z = __import__('4-sample_Z').sample_Z
"""
