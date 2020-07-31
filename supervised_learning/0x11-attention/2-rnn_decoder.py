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


class RNNDecoder(tf.keras.layers.Layer):
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
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(units=vocab,
                                       kernel_initializer='glorot_uniform')

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
        You should concatenate the context vector with x in that order
        Returns: y, s
        y is a tensor of shape (batch, vocab) containing the output word as a
        one hot vector in the target vocabulary
        s is a tensor of shape (batch, units) containing the new decoder
        hidden state
        """
        x_float = tf.to_float(x)
        attention = SelfAttention(2048)
        context, weights = attention(s_prev, hidden_states)
        # print(x.shape, context.shape)
        new_input = tf.concat([x_float, context], axis=1)
        new_input = tf.expand_dims(new_input, 1)
        # print(new_input.shape)
        outputs, s = self.gru(inputs=new_input)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        outputs = self.F(outputs)
        return (outputs, s)
