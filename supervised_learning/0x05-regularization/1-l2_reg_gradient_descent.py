#!/usr/bin/env python3
""" Function l2_gradient descent
updates the weights and biases of a neural network using gradient descent
with L2 regularization:
Y is a one-hot numpy.ndarray of shape (classes, m) that contains the correct
labels for the data
classes is the number of classes
m is the number of data points
weights is a dictionary of the weights and biases of the neural network
cache is a dictionary of the outputs of each layer of the neural network
alpha is the learning rate
lambtha is the L2 regularization parameter
L is the number of layers of the network
The neural network uses tanh activations on each layer except the last,
which uses a softmax activation
The weights and biases of the network should be updated in place
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Function 12_reg_gradient_descent"""
    dW = {}
    dWT = {}
    db = {}
    dZ = {}
    m = Y.shape[1]
    wg = weights.copy()
    posi = str(L)
    dZ['dZ'+posi] = cache['A' + posi] - Y
    db['db'+posi] = np.sum(dZ['dZ'+posi], axis=1, keepdims=True)/m
    dW['dW'+posi] = np.matmul(cache['A'+str(L - 1)],
                              dZ['dZ'+posi].T) / m
    dWT['dWT'+posi] = dW['dW'+posi].T
    weights['W'+posi] = (1+(lambtha/m))*wg['W'+posi] - alpha*dWT['dWT'+posi]
    weights['b'+posi] = (1+(lambtha/m))*wg['b'+posi] - alpha*db['db'+posi]
    for i in range(L - 1, 0, -1):
        posl = str(i-1)
        posm = str(i+1)
        pos = str(i)
        dZ['dZ'+pos] = np.matmul(wg['W'+posm].T, dZ['dZ'+posm])
        db['db'+pos] = np.sum(dZ['dZ'+pos], axis=1, keepdims=True)/m
        dW['dW'+pos] = np.matmul(cache['A'+posl],
                                 dZ['dZ'+pos].T) / m
        dWT['dWT'+pos] = dW['dW'+pos].T
        weights['W'+pos] = (1+(lambtha/m))*wg['W'+pos] - alpha*dWT['dWT'+pos]
        weights['b'+pos] = (1+(lambtha/m))*wg['b'+pos] - alpha*db['db'+pos]
    return(weights)
