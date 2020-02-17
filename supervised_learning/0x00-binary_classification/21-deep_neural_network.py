#!/usr/bin/env python3
""" Deep Neural Network class: defines a deep neural network class
with private instances attributes
"""

import numpy as np


class DeepNeuralNetwork:
    """ Class Deep Neural networks"""
    def __init__(self, nx, layers):
        """ Settings for class Deep Neural Networks"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        ly = layers.copy()
        ly.insert(0, nx)
        for l in range(1, self.__L + 1):
            if type(ly[l-1]) is not int or ly[(l-1)] < 0:
                raise TypeError("layers must be a list of positive integers")
            temp = np.random.randn(ly[l], ly[l-1]) * (np.sqrt(2/ly[l-1]))
            self.__weights['W'+str(l)] = temp
            self.__weights['b'+str(l)] = np.zeros((ly[l], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def sigmoid(self, Z):
        """Function sigmoid"""
        sigm = 1 / (1 + np.exp(-Z))
        return sigm

    def forward_prop(self, X):
        """Function forward propoagation"""
        for i in range(self.__L + 1):
            if i == 0:
                self.__cache['A0'] = X
            else:
                A_tmp = (np.matmul(self.__weights['W' + str(i)],
                                   self.__cache['A' + str(i - 1)])
                         + self.__weights['b' + str(i)])
                H_tmp = self.sigmoid(A_tmp)
                self.__cache['A' + str(i)] = H_tmp
        return (self.__cache['A'+str(self.__L)], self.__cache)

    def cost(self, Y, A):
        """Function cost"""
        m = Y.shape[1]
        num_lreg = -1 * (Y * np.log(A) + (1 - Y) *
                         np.log(1.0000001 - A))
        cost = np.sum(num_lreg)/m
        return (cost)

    def evaluate(self, X, Y):
        """Function evauate"""
        self.forward_prop(X)
        A = self.__cache['A' + str(self.__L)]
        PRED = np.where(A >= 0.5, 1, 0)
        return (PRED, self.cost(Y, A))

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Function gradient descent"""
        dW = {}
        db = {}
        dZ = {}
        m = {}
        dZ['dZ'+str(self.__L)] = 
        for i in range(self.__L, 1, -1):


        m2 = A1.shape[1]
        dZ2 = A2 - Y
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m2
        dZT2 = dZ2.T
        dW2 = np.matmul(self.__A1, dZT2) / m2
        m1 = Y.shape[1]
        dg1 = (self.__A1 * (1 - self.__A1))
        dZ1 = np.matmul(self.__W2.T, dZ2) * dg1
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m1
        dZT1 = dZ1.T
        dW1 = np.matmul(X, dZT1) / m1
        self.__W1 = self.__W1 - (alpha * dW1.T)
        self.__b1 = self.__b1 - alpha * db1
        self.__b2 = self.__b2 - alpha * db2
        self.__W2 = self.__W2 - (alpha * dW2.T)
        return()
