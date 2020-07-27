#!/usr/bin/env python3
"""
Create a class SelfAttention that inherits from tensorflow.keras.layers.Layer
to calculate the attention for machine translation based on this paper:
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
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
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

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
        mul1 = self.W(s_prev)
        mul1 = tf.expand_dims(mul1, 1)
        mul2 = self.U(hidden_states)
        weights = self.V(tf.tanh(tf.math.add(mul1, mul2)))
        outputs = tf.matmul(mul2, tf.nn.softmax(weights), transpose_a=True)
        outputs = tf.math.reduce_sum(outputs, axis=2)
        return(outputs, weights)
