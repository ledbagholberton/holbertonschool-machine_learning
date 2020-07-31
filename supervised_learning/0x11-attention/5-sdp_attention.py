#!/usr/bin/env python3
"""
Calculates the scaled dot product attention:

The preceding dimensions of Q, K, and V are the same
The preceding dimensions of mask can be broadcast into Q, K, and V
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Parameters
    ----------
    Q is a tensor with its last two dimensions as (..., seq_len_q, dk)
    containing the query matrix
    K is a tensor with its last two dimensions as (..., seq_len_k, dk)
    containing the key matrix
    V is a tensor with its last two dimensions as (..., seq_len_v, dv)
    containing the value matrix
    mask is a tensor with its last two dimens as (..., seq_len_q, seq_len_k)
    containing the optional mask, defaulted to None
    if mask is not None, multiply -1e9 to the mask and add it to the
    scaled matrix multiplication
    Returns
    -------
    output a tensor with its last two dimensions as (..., seq_len_q, dv)
    containing the scaled dot product attention
    weights a tensor with its last two dimens as (..., seq_len_q, seq_len_k)
    containing the attention weights
    """
    ten_1 = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(Q.shape[-1], dtype=tf.float32)
    den = tf.math.square(dk)
    ten_1 = ten_1 * (1 / den)
    if mask is None:
        ten_1 += ten_1 * -1e9
    ten_1 = tf.nn.softmax(ten_1)
    output = tf.matmul(ten_1,  V)
    return (output, ten_1)
