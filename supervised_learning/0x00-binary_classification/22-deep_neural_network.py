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
        dWT = {}
        db = {}
        dZ = {}
        m = {}
        wg = self.__weights.copy()
        m['m'+str(self.__L)] = self.__cache['A' + str(self.__L)].shape[1]
        dZ['dZ'+str(self.__L)] = self.__cache['A' + str(self.__L)] - Y
        db['db'+str(self.__L)] = np.sum(dZ['dZ'+str(self.__L)])/m['m'+str(self.__L)]
        for i in range(self.__L - 1, 0, -1):
            m['m'+str(i)] = self.__cache['A' + str(i + 1)].shape[1]
            mm = m['m'+str(i)]
            print("******", mm, "y i es", i)
            g_temp = self.__cache['A'+str(i)] * (1 - self.__cache['A'+str(i)])
            dZ['dZ'+str(i)] = np.matmul(wg['W'+str(i+1)].T, dZ['dZ'+str(i+1)]) - g_temp
            db['db'+str(i)] = np.sum(dZ['dZ'+str(i)])/mm
            dW['dW'+str(i)] = np.matmul(self.__cache['A'+str(i-1)], dZ['dZ'+str(i)].T) / mm
            dWT['dWT'+str(i)] = dW['dW'+str(i)].T
            self.__weights['W'+str(i)] = wg['W'+str(i)] - alpha*dWT['dWT'+str(i)]
            self.__weights['b'+str(i)] = wg['b'+str(i)] - alpha*db['db'+str(i)]
        return()

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Function train"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        PRED, cost = self.evaluate(X, Y)
        for i in range(iterations):
            self.gradient_descent(X, Y, self.__cache, alpha)
            PRED, cost = self.evaluate(X, Y)
        return(PRED, cost)