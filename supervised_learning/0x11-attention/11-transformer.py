#!/usr/bin/env python3
"""
Create a class Transformer that inherits from tensorflow.keras.Model to create a transformer network:

Class constructor 

Sets the following public instance attributes:
encoder - the encoder layer
decoder - the decoder layer
linear - a final Dense layer with target_vocab units
Public instance method def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):
inputs - a tensor of shape (batch, input_seq_len, dm)containing the inputs
target - a tensor of shape (batch, target_seq_len, dm)containing the target
training - a boolean to determine if the model is training
encoder_mask - the padding mask to be applied to the encoder
look_ahead_mask - the look ahead mask to be applied to the decoder
decoder_mask - the padding mask to be applied to the decoder
Returns: a tensor of shape (batch, target_seq_len, target_vocab) containing the transformer output
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


Class Transformer()
    """
    """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Parameters
        ----------
        N - the number of blocks in the encoder and decoder
        dm - the dimensionality of the model
        h - the number of heads
        hidden - the number of hidden units in the fully connected layers
        input_vocab - the size of the input vocabulary
        target_vocab - the size of the target vocabulary
        max_seq_input - the maximum sequence length possible for the input
        max_seq_target - the maximum sequence length possible for the target
        drop_rate - the dropout rate. Default = 0.1

        Returns
        -------
        None.

        """
        self.encoder = 
        self.decoder = 
        self.linear = tf.keras.layers.Dense(units=vocab,
                                            kernel_initializer='glorot_uniform')
        
     def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
              decoder_mask):
        """
        Public instance method def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):
        inputs - a tensor of shape (batch, input_seq_len, dm)containing the inputs
        target - a tensor of shape (batch, target_seq_len, dm)containing the target
        training - a boolean to determine if the model is training
        encoder_mask - the padding mask to be applied to the encoder
        look_ahead_mask - the look ahead mask to be applied to the decoder
        decoder_mask - the padding mask to be applied to the decoder
        Returns: a tensor of shape (batch, target_seq_len, target_vocab) containing the transformer output

         Parameters
         ----------
         inputs : TYPE
             DESCRIPTION.
         target : TYPE
             DESCRIPTION.
         training : TYPE
             DESCRIPTION.
         encoder_mask : TYPE
             DESCRIPTION.
         look_ahead_mask : TYPE
             DESCRIPTION.
         decoder_mask : TYPE
             DESCRIPTION.

         Returns
         -------
         None.

         """