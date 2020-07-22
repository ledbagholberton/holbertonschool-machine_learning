#!/usr/bin/env python3
"""
Create a class RNNEncoder that inherits from tensorflow.keras.layers.Layer to
encode for machine translation:

 


Public instance method def call(self, x, initial):
x is a tensor of shape (batch, input_seq_len) containing the input to the
encoder layer as word indices within the vocabulary
initial is a tensor of shape (batch, units) containing the initial hidden state
Returns: outputs, hidden
outputs is a tensor of shape (batch, input_seq_len, units)containing the
outputs of the encoder
hidden is a tensor of shape (batch, units) containing the last hidden state
of the encoder
"""
import tensorflow as tf


Class RNNEncoder(tensorflow.keras.layers.Layer):
    """Class RNNEncoder"""
    def __init__(self, vocab, embedding, units, batch):
        """Constructor for Class RNNEncoder
        Parameters:
        vocab is an integer representing the size of the input vocabulary
        embedding is an integer representing the dimensionality of the
        embedding vector
        units is an integer representing the number of hidden units in the
        RNN cell
        batch is an integer representing the batch size
        Actions:
        Sets the following public instance attributes:
        batch - the batch size
        units - the number of hidden units in the RNN cell
        embedding - a keras Embedding layer that converts words from the
        vocabulary into an embedding vector
        gru - a keras GRU layer with units units
        Outputs:
        Should return both the full sequence of outputs as well as the last
        hidden state
        Recurrent weights should be initialized with glorot_uniform
        """
        self.batch = batch
        self.units = units
        self.embedding = embedding
        self.gru = 
        
    def initialize_hidden_state(self):
        """
        Public instance method def initialize_hidden_state(self):
        Initializes the hidden states for the RNN cell to a tensor of zeros
        Returns: a tensor of shape (batch, units)containing the initialized
        hidden states
        """
    def call(self, x, initial):
        """
        x is a tensor of shape (batch, input_seq_len) containing the input to 
        the encoder layer as word indices within the vocabulary
        initial is a tensor of shape (batch, units) containing the initial
        hidden state
        Returns: outputs, hidden
        outputs is a tensor of shape (batch, input_seq_len, units)
        containing the Outputs of the encoder
        hidden is a tensor of shape (batch, units) containing the last hidden
        state Of the encoder
        """
        
    
    
