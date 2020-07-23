#!/usr/bin/env python3
"""
Sets the following public instance attributes:
embedding - a keras Embedding layer that converts words from the vocabulary
into an embedding vector
gru - a keras GRU layer with units units
Should return both the full sequence of outputs as well as the last hidden
state
Recurrent weights should be initialized with glorot_uniform
F - a Dense layer with vocab units

"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


Class RNNDecoder(tensorflow.keras.layers.Layer):
    """Class RNNDecoder"""
    def __init__(self, vocab, embedding, units, batch):
        """
        Parameters
        ----------
        vocab is an integer representing the size of the output vocabulary
        embedding is an integer representing the dimensionality of the
        embedding vector
        units is an integer representing the number of hidden units in
        the RNN cell
        batch is an integer representing the batch size

        Returns
        -------
        None.

        """
        self.embedding = embedding
        self.gru = tf.keras.layers.GRU(units = self.units,
                                       kernel_initializer=glorot_uniform)
        self.F = tf.keras.layers.Dense(units=vocab,
                                       kernel_initializer='glorot_uniform')
        return()
    
    def call(self, x, s_prev, hidden_states):
        """
        Parameters
        Public instance method
        x is a tensor of shape (batch, target_seq_len) containing the input
        to the decoder layer as word indices within the vocabulary
        s_prev is a tensor of shape (batch, units) containing the previous
        decoder hidden state
        hidden_states is a tensor of shape (batch, target_seq_len, units)
        containing the outputs of the decoder
        """
        