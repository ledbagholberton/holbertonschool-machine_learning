#!/usr/bin/env python3
"""
class DecoderBlock
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Class Decoder Block"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Constructor
        Class constructor
        dm - the dimensionality of the model
        h - the number of heads
        hidden - the number of hidden units in the fully connected layer
        drop_rate - the dropout rate
        Sets the following public instance attributes:
        mha1 - the first MultiHeadAttention layer
        mha2 - the second MultiHeadAttention layer
        dense_hidden - hidden dense layer with hidden units /relu activation
        dense_output - the output dense layer with dm units
        layernorm1 - the first layer norm layer, with epsilon=1e-6
        layernorm2 - the second layer norm layer, with epsilon=1e-6
        layernorm3 - the third layer norm layer, with epsilon=1e-6
        dropout1 - the first dropout layer
        dropout2 - the second dropout layer
        dropout3 - the third dropout layer
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        x - a tensor of shape (batch, target_seq_len, dm)containing the input
        to the decoder block
        encoder_output - a tensor of shape (batch, input_seq_len, dm)containing
        the output of the encoder
        training - a boolean to determine if the model is training
        look_ahead_mask - the mask to be applied to the first multi head
        attention layer
        padding_mask - the mask to be applied to the second multi head
        attention layer
        Returns: a tensor of shape (batch, target_seq_len, dm) containing the
        block’s output
        """
        a, _ = self.mha1(x, x, x, mask=look_ahead_mask)
        b = self.dropout1(a, training=training)
        c = self.layernorm1(b + x)
        d, _ = self.mha2(c, encoder_output, encoder_output, mask=padding_mask)
        e = self.dropout2(d, training=training)
        f = self.layernorm2(e + c)
        g = self.dense_hidden(f)
        h = self.dense_output(g)
        i = self.dropout2(h, training=training)
        j = self.layernorm2(i + f)
        return j
