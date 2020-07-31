#!/usr/bin/env python3
"""
Create a  that inherits from tensorflow.keras.layers.Layer to perform
multi head attention

Public instance method
Q is a tensor of shape (batch, seq_len_q, dk) containing:
the input to generate the query matrix
K is a tensor of shape (batch, seq_len_k, dk) containing:
the input to generate the key matrix
V is a tensor of shape (batch, seq_len_v, dv) containing:
the input to generate the value matrix
mask is always None
Returns: output, weights
outputa tensor with its last two dimensions as
(..., seq_len_q, dv) containing the scaled dot product attention
weights a tensor with its last two dimensions as
(..., seq_len_q, seq_len_k) containing the attention weights
You can use sdp_attention = __import__('5-sdp_attention').sdp_attention
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Class MultiHeadAttention"""
    def __init__(self, dm, h):
        """Constructor
        dm is an integer representing the dimensionality of the model
        h is an integer representing the number of heads
        dm is divisible by h
        Sets the following public instance attributes:
        h - the number of heads
        dm - the dimensionality of the model
        depth - the depth of each attention head
        Wq - a Dense layer with dm units, used to generate the query matrix
        Wk - a Dense layer with dm units, used to generate the key matrix
        Wv - a Dense layer with dm units, used to generate the value matrix
        linear - a Dense layer with dm units, used to generate attention output
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm / h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def call(self, Q, K, V, mask):
        """
        Q is a tensor of shape (batch, seq_len_q, dk) containing
        the input to generate the query matrix
        K is a tensor of shape (batch, seq_len_k, dk) containing
        the input to generate the key matrix
        V is a tensor of shape (batch, seq_len_v, dv) containing
        the input to generate the value matrix
        mask is always None
        Returns: output, weights
        outputa tensor with its last two dimensions as (..., seq_len_q, dv)
        the scaled dot product attention
        weights a tensor with its last two dimensions as
        (..., seq_len_q, seq_len_k) containing the attention weights
        """
        batch, seq_len_q, dk = Q.shape
        seq_len_k = K.shape[1]
        _, seq_len_v, dv = V.shape
        # paso los vectores por las capas definidas en el constructor
        tQ = self.Wq(Q)
        tK = self.Wk(K)
        tV = self.Wv(V)
        # paso a traves de las linear del constructor
        tQ_len = self.linear(tQ)
        tK_len = self.linear(tK)
        tV_len = self.linear(tV)
        # cambio el shape e intercambio las posiciones 1 y 2 para
        # acomodarlo a las dimensiones requeridas por SDP
        new_shape_Q = (self.h, -1, seq_len_q, dk//self.h)
        tQ_1 = tf.reshape(tQ_len, new_shape_Q)
        tQ_2 = tf.transpose(tQ_1, perm=[0, 2, 1, 3])
        new_shape_K = (self.h, -1, seq_len_k, dk//self.h)
        tK_1 = tf.reshape(tK_len, new_shape_K)
        tK_2 = tf.transpose(tK_1, perm=[0, 2, 1, 3])
        new_shape_V = (self.h, -1, seq_len_v, dv//self.h)
        tV_1 = tf.reshape(tV_len, new_shape_V)
        tV_2 = tf.transpose(tV_1, perm=[0, 2, 1, 3])
        # (tQ_2.shape, tK_2.shape, tV_2.shape)
        # (8, 15, 100, 32) (8, 15, 100, 32) (8, 15, 100, 32)
        # lo paso por el SDP + head
        output, weights = sdp_attention(tQ_2, tK_2, tV_2)
        # (output.shape, weights.shape)
        # (8, 15, 100, 32) (8, 15, 100, 100)
        # vuelvo a arreglar el output - concatenando los head.
        output_t = tf.transpose(output, perm=[0, 2, 1, 3])
        output_c = tf.reshape(output_t, (batch, -1, self.dm))
        # print(output_c.shape, weights.shape)
        # (50, 15, 512) (8, 15, 100, 100)
        output_l = self.linear(output_c)
        return (output_l, weights)
