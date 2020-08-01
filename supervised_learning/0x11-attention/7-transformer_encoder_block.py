#!/usr/bin/env python3
"""
class EncoderBlock
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Class MultiHeadAttention"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Constructor
        Class constructor
        dm - the dimensionality of the model
        h - the number of heads
        hidden - the number of hidden units in the fully connected layer
        drop_rate - the dropout rate
        Sets the following public instance attributes:
        mha - a MultiHeadAttention layer
        dense_hidden - the hidden dense layer with hidden units and relu
        activation
        dense_output - the output dense layer with dm units
        layernorm1 - the first layer norm layer, with epsilon=1e-6
        layernorm2 - the second layer norm layer, with epsilon=1e-6
        dropout1 - the first dropout layer
        dropout2 - the second dropout layer
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        x - a tensor of shape (batch, input_seq_len, dm)containing the input
        to the encoder block
        training - a boolean to determine if the model is training
        mask - the mask to be applied for multi head attention
        Returns: a tensor of shape (batch, input_seq_len, dm) containing
        the block’s output
        """
        a, _ = self.mha(x, x, x, mask)
        b = self.dropout1(a, training=training)
        c = self.layernorm1(a + x)
        d = self.dense_hidden(c)
        e = self.dense_output(d)
        f = self.dropout2(e, training=training)
        g = self.layernorm2(f + c)
        return g
