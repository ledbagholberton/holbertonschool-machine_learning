#!/usr/bin/env python3
"""


"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """
    class Transformer
    """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Parameters
        ----------
        encoder - the encoder layer
        decoder - the decoder layer
        linear - a final Dense layer with target_vocab units
        Returns
        -------
        None.
        """
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input,
                               drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target,
                               drop_rate)
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        inputs - a tensor of shape (batch, input_seq_len, dm) containing
        the inputs
        target - a tensor of shape (batch, target_seq_len, dm)containing
        the target
        training - a boolean to determine if the model is training
        encoder_mask - the padding mask to be applied to the encoder
        look_ahead_mask - the look ahead mask to be applied to the decoder
        decoder_mask - the padding mask to be applied to the decoder
        Returns: a tensor of shape (batch, target_seq_len, target_vocab)
        containing the transformer output
        """
        encoder_output = self.encoder(inputs,
                                      training=training,
                                      mask=encoder_mask)
        decoder_output = self.decoder(target, encoder_output,
                                      training=training,
                                      look_ahead_mask, decoder_mask)
        output = self.linear(decoder_output)
        return output
