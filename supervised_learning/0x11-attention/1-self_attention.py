#!/usr/bin/env python3
"""
Create a class SelfAttention that inherits from tensorflow.keras.layers.Layer
to calculate the attention for machine translation based on this paper:

Class constructor 

Public instance method 

Returns: 

"""
import tensorflow as tf


Class SelfAttention(tensorflow.keras.layers.Layer):
    """Class RNNEncoder"""
    def __init__(self, units):
        """ Constructor
        units is an integer representing the number of hidden units in the
        alignment model
        Sets the following public instance attributes:
        W - a Dense layer with units units, to be applied to the previous
        decoder hidden state
        U - a Dense layer with units units, to be applied to the encoder
        hidden states
        V - a Dense layer with 1 units, to be applied to the tanh of the sum
        of the outputs of W and U
        """
        self.W = tf.keras.layers.Dense(units=units,
                                       kernel_initializer='glorot_uniform')
        self.U = tf.keras.layers.Dense(units=units,
                                       kernel_initializer='glorot_uniform')
        self.V = tf.keras.layers.Dense(units=1,
                                       kernel_initializer='glorot_uniform')
    
    def call(self, s_prev, hidden_states):
        """
        Parameters
        ----------
        s_prev is a tensor of shape (batch, units) containing the previous
        decoder hidden state
        hidden_states is a tensor of shape (batch, input_seq_len, units)
        containing the outputs of the encoder

        Returns
        -------
        context is a tensor of shape (batch, units) that contains the context
        vector for the decoder
        weights is a tensor of shape (batch, input_seq_len, 1) that contains
        the attention weights
        """
        