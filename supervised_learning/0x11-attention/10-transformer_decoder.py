#!/usr/bin/env python3
"""
class Decoder
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Class Decoder"""
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Parameters
        ----------
        N - the number of blocks in the encoder
        dm - the dimensionality of the model
        h - the number of heads
        hidden - the number of hidden units in the fully connected layer
        target_vocab - the size of the target vocabulary
        max_seq_len - the maximum sequence length possible
        drop_rate - the dropout rate
        Sets the following public instance attributes:
        N - the number of blocks in the encoder
        dm - the dimensionality of the model
        embedding - the embedding layer for the targets
        positional_encoding - a numpy.ndarray of shape (max_seq_len, dm)
        containing the positional encodings
        blocks - a list of length N containing all of the DecoderBlock‘s
        dropout - the dropout layer, to be applied to the positional encodings
        Returns
        -------
        None
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)] * N
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """
        Parameters
        ----------
        Public instance method
        x - a tensor of shape (batch, target_seq_len, dm)containing the input
        to the decoder
        encoder_output - a tensor of shape (batch, input_seq_len, dm)containing
        the output of the encoder
        training - a boolean to determine if the model is training
        look_ahead_mask - the mask to be applied to the first multi head
        attention layer
        padding_mask - the mask to be applied to the second multi head
        attention layer

        Returns
        -------
        A tensor of shape (batch, target_seq_len, dm) containing the
        decoder output
        """
        seq_len = x.shape[1]
        # embbeding to input x
        x = self.embedding(x)
        # scale x with sqr of dm
        x = x * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        # sum of positional encoding
        x = x + self.positional_encoding[:seq_len]
        # dropout to everyhting
        output = self.dropout(x, training=training)
        for i in range(self.N):
            output = self.blocks[i](output, encoder_output, training,
                                    look_ahead_mask, padding_mask)
        return output
