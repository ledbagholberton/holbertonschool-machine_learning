#!/usr/bin/env python3
"""
Calculates the scaled dot product attention:

The preceding dimensions of Q, K, and V are the same
The preceding dimensions of mask can be broadcast into Q, K, and V
"""
import tensoflow as tf


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
    outputa tensor with its last two dimensions as (..., seq_len_q, dv)
    containing the scaled dot product attention
    weights a tensor with its last two dimens as (..., seq_len_q, seq_len_k)
    containing the attention weights

    """
    