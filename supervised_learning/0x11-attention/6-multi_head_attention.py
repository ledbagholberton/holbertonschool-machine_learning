#!/usr/bin/env python3
"""
Create a class MultiHeadAttention that inherits from tensorflow.keras.layers.Layer to perform multi head attention:

Class constructor def __init__(self, dm, h):
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
linear - a Dense layer with dm units, used to generate the attention output
Public instance method def call(self, Q, K, V, mask):
Q is a tensor of shape (batch, seq_len_q, dk) containing the input to generate the query matrix
K is a tensor of shape (batch, seq_len_k, dk) containing the input to generate the key matrix
V is a tensor of shape (batch, seq_len_v, dv) containing the input to generate the value matrix
mask is always None
Returns: output, weights
outputa tensor with its last two dimensions as (..., seq_len_q, dv) containing the scaled dot product attention
weights a tensor with its last two dimensions as (..., seq_len_q, seq_len_k) containing the attention weights
You can use sdp_attention = __import__('5-sdp_attention').sdp_attention
