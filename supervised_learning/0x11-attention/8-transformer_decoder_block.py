#!/usr/bin/env python3
"""
Create a class DecoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer:

Class constructor def __init__(self, dm, h, hidden, drop_rate=0.1):
dm - the dimensionality of the model
h - the number of heads
hidden - the number of hidden units in the fully connected layer
drop_rate - the dropout rate
Sets the following public instance attributes:
mha1 - the first MultiHeadAttention layer
mha2 - the second MultiHeadAttention layer
dense_hidden - the hidden dense layer with hidden units and relu activation
dense_output - the output dense layer with dm units
layernorm1 - the first layer norm layer, with epsilon=1e-6
layernorm2 - the second layer norm layer, with epsilon=1e-6
layernorm3 - the third layer norm layer, with epsilon=1e-6
dropout1 - the first dropout layer
dropout2 - the second dropout layer
dropout3 - the third dropout layer
Public instance method def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
x - a tensor of shape (batch, target_seq_len, dm)containing the input to the decoder block
encoder_output - a tensor of shape (batch, input_seq_len, dm)containing the output of the encoder
training - a boolean to determine if the model is training
look_ahead_mask - the mask to be applied to the first multi head attention layer
padding_mask - the mask to be applied to the second multi head attention layer
Returns: a tensor of shape (batch, target_seq_len, dm) containing the block’s output
You can use MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention
