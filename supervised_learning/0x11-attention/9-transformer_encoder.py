#!/usr/bin/env python3
"""
Class Encoder
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Class Encoder"""
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Parameters
        ----------
        N - the number of blocks in the encoder
        dm - the dimensionality of the model
        h - the number of heads
        hidden - the number of hidden units in the fully connected layer
        input_vocab - the size of the input vocabulary
        max_seq_len - the maximum sequence length possible
        drop_rate - the dropout rate
        Sets the following public instance attributes:
        N - the number of blocks in the encoder
        dm - the dimensionality of the model
        embedding - the embedding layer for the inputs
        positional_encoding - a numpy.ndarray of shape (max_seq_len, dm)
        containing the positional encodings
        blocks - a list of length N containing all of the EncoderBlock‘s
        dropout - the dropout layer, to be applied to the positional encodings

        Returns
        -------
        None.
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)] * N
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Parameters
        ----------
        x - a tensor of shape (batch, input_seq_len, dm)containing the input
        to the encoder
        training - a boolean to determine if the model is training
        mask - the mask to be applied for multi head attention
        Returns:

        Returns
        -------
        A tensor of shape (batch, input_seq_len, dm) containing encoder output
        """
        seq_len = tf.shape(x)[1]
        pos = tf.cast(self.positional_encoding, tf.float32)
        input_enc = self.embedding(x)
        input_scale = input_enc / tf.math.sqrt(tf.cast(self.dm, tf.float32))
        input_pos = input_scale + pos[:seq_len]
        output = self.dropout(input_pos, training=training)
        for i in range(1, self.N):
            output = self.blocks[i](output, training=training, mask=mask)
        return output
