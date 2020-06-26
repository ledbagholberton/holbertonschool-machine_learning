#!/usr/bin/env python3
"""
Write a function  that creates a variational autoencoder:

input_dims is an integer containing the dimensions of the model input
hidden_layers is a list containing
the number of nodes for each hidden layer in the encoder, respectively
the hidden layers should be reversed for the decoder
latent_dims is an integer containing the dimensions of the latent space
representation
Returns: encoder, decoder, auto
encoder is the encoder model, which should output the latent representation,
the mean, and the log variance, respectively
decoder is the decoder model
auto is the full autoencoder model
The autoencoder model should be compiled using:
adam optimization and binary cross-entropy loss
All layers should use a relu activation except:
-for the mean and log variance layers in the encoder,which should use None,
and the last layer in the decoder, which should use sigmoid
"""
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder"""

    # VAE model = encoder + decoder
    # build encoder model
    size = len(hidden_layers)
    x = K.layers.Input(shape=input_dims, name='encoder_input')
    inputs = x
    for i in range(size):
        x = K.layers.Dense(hidden_layers[i], activation='relu')(x)
    z_mean = K.layers.Dense(latent_dims, name='z_mean')(x)
    z_log_var = K.layers.Dense(latent_dims, name='z_log_var')(x)
    z = K.layers.Lambda(sampling, output_shape=(latent_dims,), name='z')([z_mean, z_log_var])
    encoder = K.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # build decoder model
    y = K.layers.Input(shape=(latent_dims,), name='z_sampling')
    latent_inputs = y
    for i in range(size, 0, -1):
        y = K.layers.Dense(hidden_layers[i-1], activation='relu')(y)
    y = K.layers.Dense(input_dims, activation='sigmoid')(y)
    decoder = K.models.Model(latent_inputs, y, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    auto = K.models.Model(inputs, outputs, name='auto')
    auto.summary()    
    auto.compile(optimizer='Adam', loss=loss_autoencoder)
    return(encoder, decoder, auto)


def loss_autoencoder(z_mean, z_log_var):
    reconstruction_loss = K.losses.binary_crossentropy(z_mean, z_log_var)
    kl_loss = 1 + z_log_var - K.backend.square(z_mean) - K.backend.exp(z_log_var)
    kl_loss = K.backend.sum(kl_loss, axis=-1)
    kl_loss *= kl_loss * (-0.5)
    total_loss = K.backend.mean(reconstruction_loss + kl_loss)
    return total_loss

def sampling(args):
    """Sampling of normal distribution with z_mean & z_log_var"""
    z_mean, z_log_var = args
    batch = K.backend.shape(z_mean)[0]
    print("Este es BATCH", batch)
    dim = K.backend.int_shape(z_mean)[1]
    print("Este es DIM", dim)
    epsilon = K.backend.random_normal(shape=(batch, dim))
    return z_mean + K.backend.exp(0.5 * z_log_var) * epsilon
