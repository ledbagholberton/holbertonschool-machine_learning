"""
Function  that conducts forward propagation using Dropout:
X is a numpy.ndarray of shape (nx, m) containing the input data for the network
nx is the number of input features
m is the number of data points
weights is a dictionary of the weights and biases of the neural network
L the number of layers in the network
keep_prob is the probability that a node will be kept
All layers except the last should use the tanh activation function
The last layer should use the softmax activation function
Returns: a dictionary containing the outputs of each layer and the dropout mask
used on each layer (see example for format)
"""
import tensorflow as tf
import numpy as np


def tanh(Z):
    """Function tanh"""
    tanh = np.tanh(Z)
    return tanh


def softmax(Z):
    """Function softmax"""
    t = np.exp(Z)
    sum = np.sum(t, axis=0, keepdims=True)
    softmax = t / sum
    return softmax


def dropout_forward_prop(X, weights, L, keep_prob):
    """FUnction Dropout Forward Prop"""
    cache = {}
    for i in range(L):
        if i == 0:
            cache['A0'] = X
        else:
            A_tmp_0 = np.matmul(weights['W' + str(i)],
                                cache['A' + str(i-1)])
            A_tmp = A_tmp_0 + weights['b' + str(i)]
            cache['D' + str(i)] = np.random.rand(A_tmp.shape[0],
                                                 A_tmp.shape[1])
            cache['D' + str(i)] = cache['D' + str(i)] < keep_prob
            A_tmp = np.multiply(A_tmp, cache['D' + str(i)])
            A_tmp = A_tmp / keep_prob
            if i == L:
                H_tmp = softmax(A_tmp)
            else:
                H_tmp = tanh(A_tmp)
            cache['A' + str(i)] = H_tmp
    return (cache)
