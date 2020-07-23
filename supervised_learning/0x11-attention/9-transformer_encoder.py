#!/usr/bin/env python3
"""
Create a class Encoder that inherits from tensorflow.keras.layers.Layer to create the encoder for a transformer:

Class constructor def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
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
positional_encoding - a numpy.ndarray of shape (max_seq_len, dm) containing the positional encodings
blocks - a list of length N containing all of the EncoderBlock‘s
dropout - the dropout layer, to be applied to the positional encodings
Public instance method call(self, x, training, mask):
x - a tensor of shape (batch, input_seq_len, dm)containing the input to the encoder
training - a boolean to determine if the model is training
mask - the mask to be applied for multi head attention
Returns: a tensor of shape (batch, input_seq_len, dm) containing the encoder output
You can use positional_encoding = __import__('4-positional_encoding').positional_encoding and EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock
