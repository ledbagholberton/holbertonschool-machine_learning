#!/usr/bin/env python3
"""
Positional Encoding
"""
import tensorflow as tf
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates de positional encoding for a transformer
    Parameters
    ----------
    max_seq_len is an integer representing the maximum sequence length
    dm is the model depth
    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing the
    positional encoding vectors
    """
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dm) for j in range(dm)]
        if pos != 0 else np.zeros(dm) for pos in range(max_seq_len)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
    return position_enc
