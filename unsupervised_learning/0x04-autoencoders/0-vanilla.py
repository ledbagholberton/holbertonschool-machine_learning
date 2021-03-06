"""
input_dims is an integer containing the dimensions of the model input
hidden_layers is a list containing the number of nodes for each hidden
layer in the encoder, respectively
the hidden layers should be reversed for the decoder
latent_dims is an integer containing the dimensions of the
latent space representation
Returns: encoder, decoder, auto
encoder is the encoder model
decoder is the decoder model
auto is the full autoencoder model
The autoencoder model should be compiled using adam optimization and
binary cross-entropy loss
All layers should use a relu activation except for the last layer in
the decoder, which should use sigmoid
"""
import tensorflow.keras as K

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Autoencoder Vanilla
    """
    nodes = len(hidden_layers)
    input_img = K.layers.Input(shape=(input_dims,))
    encoded = K.layers.Dense(hidden_layers[0], activation='relu')(input_img)
    for i in range(1, nodes):
        encoded = K.layers.Dense(hidden_layers[i], activation='relu')(encoded)
    lay_latent = K.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = K.models.Model(input_img, lay_latent)
    # "decoded" is the lossy reconstruction of the input
    input_dec = K.layers.Input(shape=(latent_dims,))
    decoded = K.layers.Dense(hidden_layers[i], activation='relu')(input_dec)
    for i in range(nodes-2, 0, -1):
        decoded = K.layers.Dense(hidden_layers[i], activation='relu')(decoded)
    decoded = K.layers.Dense(input_dims, activation='sigmoid')(decoded)
    # this model maps an input to its reconstruction
    decoder = K.models.Model(input_dec, decoded)
    enc_out = encoder(input_img)
    dec_out = decoder(enc_out)
    auto = K.models.Model(input_img, dec_out)
    auto.compile(optimizer='Adam', loss='binary_crossentropy')
    return(encoder, decoder, auto)
