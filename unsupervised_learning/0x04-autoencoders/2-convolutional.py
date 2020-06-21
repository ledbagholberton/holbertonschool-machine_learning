"""
input_dims is a tuple of integers containing the dimensions of the model input
filters is a list containing the number of filters for each convolutional
layer in the encoder, respectively
the filters should be reversed for the decoder
latent_dims is a tuple of integers containing the dimensions of the latent
space representation
Each convolution in the encoder should use a kernel size of (3, 3) with same
padding and relu activation, followed by max pooling of size (2, 2)
Each convolution in the decoder, except for the last two, should use a
filter size of (3, 3) with same padding and relu activation, followed by
upsampling of size (2, 2)
The second to last convolution should instead use valid padding
The last convolution should have only 1 filter with sigmoid activation
and no upsampling
Returns: encoder, decoder, auto
encoder is the encoder model
decoder is the decoder model
auto is the full autoencoder model
The autoencoder model should be compiled using adam optimization and
binary cross-entropy loss
"""
import tensorflow.keras as K
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


def autoencoder(input_dims, filters, latent_dims):
    """
    Convolutional Autoencoder
    """
    size = len(filters)
    input_img = K.layers.Input(shape=input_dims)
    x = Conv2D(filters[0], (3, 3), activation='relu',
               padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    for i in range(1, size-2):
        x = Conv2D(filters[i], (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(filters[size-1], (3, 3), activation='relu', padding='same')(x)
    lay_latent = MaxPooling2D((2, 2), padding='same')(x)
    encoder = K.models.Model(input_img, lay_latent)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    input_dec = K.layers.Input(shape=latent_dims)
    y = Conv2D(8, (3, 3), activation='relu', padding='same')(input_dec)
    y = UpSampling2D((2, 2))(y)
    for i in range(size-2, 0, -1):
        y = Conv2D(filters[i], (3, 3), activation='relu', padding='same')(y)
        y = MaxPooling2D((2, 2), padding='same')(y)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(y)
    decoder = K.models.Model(input_dec, decoded)
    enc_out = encoder(input_img)
    dec_out = decoder(enc_out)
    auto = K.models.Model(input_img, dec_out)
    auto.compile(optimizer='Adam', loss='binary_crossentropy')
    return(encoder, decoder, auto)
