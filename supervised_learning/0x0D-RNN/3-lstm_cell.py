#!/usr/bin/env python3
"""
Creates the public instance attributes
Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by that represent
the weights and biases of the cell
Wfand bf are for the forget gate
Wuand bu are for the update gate
Wcand bc are for the intermediate cell state
Woand bo are for the output gate
Wyand by are for the outputs
The weights should be initialized using a random normal distribution
in the order listed above
The weights will be used on the right side for matrix multiplication
The biases should be initialized as zeros
"""
import numpy as np


class LSTMCell:
	"""Class RNNCell"""
	def __init__(self, i, h, o):
		""" Constructor
		i is the dimensionality of the data
		h is the dimensionality of the hidden state
		o is the dimensionality of the outputs
		"""
		self.Wf = np.random.randn(h + i, h)
		self.Wu = np.random.randn(h + i, h)
		self.Wc = np.random.randn(h + i, h)
		self.Wo = np.random.randn(h + i, h)
		self.Wy = np.random.randn(h, o)
		self.bf = np.zeros((1, h))
		self.bu = np.zeros((1, h))
		self.bc = np.zeros((1, h))
		self.bo = np.zeros((1, h))
		self.by = np.zeros((1, o))

	def forward(self, h_prev, c_prev, x_t):
		""" Method Forward		
        performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains:
		the data input for the cell
        m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing:
		the previous hidden state
        c_prev is a numpy.ndarray of shape (m, h) containing:
		the previous cell state
        Returns: h_next, c_next, y
        h_next is the next hidden state
        c_next is the next cell state
        y is the output of the cell
		"""
		m, i = x_t.shape
		_, h = h_prev.shape
		st_ct_1 = np.hstack((h_prev, x_t))
		g_u = self.sigmoid(np.matmul(st_ct_1, self.Wu) + self.bu)
		g_f = self.sigmoid(np.matmul(st_ct_1, self.Wf) + self.bf)
		g_o = self.sigmoid(np.matmul(st_ct_1, self.Wo) + self.bo)
		c_tilde = np.tanh(np.matmul(st_ct_1, self.Wc) + self.bc)
		c_next = (g_u * c_tilde) + (g_f * c_prev)
		h_next = g_o * np.tanh(c_next)
		y_n = np.matmul(h_next, self.Wy) + self.by
		y = self.softmax(y_n)
		return (h_next, c_next, y)
        
	def softmax(self, X):
		expo = np.exp(X)
		expo_sum = np.sum(np.exp(X), axis=-1, keepdims=True)
		return expo/expo_sum
	
	def sigmoid(self, X):
		return 1/(1 + np.exp(-X))
