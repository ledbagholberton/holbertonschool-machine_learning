"""
updates the weights of a neural network with Dropout regularization
using gradient descent:

Y is a one-hot numpy.ndarray of shape (classes, m) that contains the correct
labels for the data:
classes is the number of classes
m is the number of data points
weights is a dictionary of the weights and biases of the neural network
cache is a dictionary of the outputs and dropout masks of each layer of
the neural network
alpha is the learning rate
keep_prob is the probability that a node will be kept
L is the number of layers of the network
All layers use the tanh activation function except the last, which uses the softmax
activation function
The weights of the network should be updated in place
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


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """FUnction Dropout Gradient Descent"""
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
            cache['D' + str(i)] = int(cache['D' + str(i)] == 'true')
            A_tmp = np.multiply(A_tmp, cache['D' + str(i)])
            A_tmp = A_tmp / keep_prob
            if i == L:
                H_tmp = softmax(A_tmp)
            else:
                H_tmp = tanh(A_tmp)
            cache['A' + str(i)] = H_tmp
    return (cache)
