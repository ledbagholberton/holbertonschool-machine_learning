#!/usr/bin/env python3
""" Neuron class: defines a single neuron performing binary classification
with pivate instances attributes
"""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """ Class Neuron """
    def __init__(self, nx):
        """ Settings for class Neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def sigmoid(self, Z):
        """function sigmoid"""
        sigm = 1 / (1 + np.exp(-Z))
        return sigm

    def forward_prop(self, X):
        """function forward_prop"""
        recta = np.matmul(self.W, X) + self.b
        H = self.sigmoid(recta)
        self.__A = H
        return (self.__A)

    def log_reg(self, y_r, y_p):
        """function log_reg"""
        num_lreg = -1 * (y_r * np.log(y_p) + (1 - y_r) *
                         np.log(1.0000001 - y_p))
        return (num_lreg)

    def cost(self, Y, A):
        """function cost"""
        m = Y.shape[1]
        num_lreg = -1 * (Y * np.log(A) + (1 - Y) *
                         np.log(1.0000001 - A))
        cost = np.sum(num_lreg)/m
        return (cost)

    def evaluate(self, X, Y):
        """function evaluate"""
        self.forward_prop(X)
        PRED = np.where(self.__A >= 0.5, 1, 0)
        return (PRED, self.cost(Y, self.__A))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """function gradient_descent"""
        m = Y.shape[1]
        Z = A - Y
        db = np.sum(Z) / m
        self.__b = self.__b - alpha * db
        ZT = Z.T
        dW = np.matmul(X, ZT) / m
        self.__W = self.__W - (alpha * dW.T)
        return()

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """function train"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        arr_cost = []
        arr_pos = []
        PRED, cost = self.evaluate(X, Y)
        iters = 0
        pos = 0
        arr_pos.append(0)
        arr_cost.append(cost)
        if verbose is True:
            print("Cost after {} iterations: {}".format(0, cost))
        for i in range(1, iterations):
            self.gradient_descent(X, Y, self.__A, alpha)
            PRED, cost = self.evaluate(X, Y)
            iters = iters + 1
            if iters == step:
                iters = 0
                pos = pos + 1
                arr_pos.append(i)
                arr_cost.append(cost)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
        PRED, cost = self.evaluate(X, Y)
        pos = pos + 1
        arr_pos.append(iterations)
        arr_cost.append(cost)
        if verbose is True:
            print("Cost after {} iterations: {}".format(iterations, cost))
        if graph is True:
            plt.xlim(0, iterations)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Training cost")
            plt.plot(arr_pos, arr_cost)
            plt.show()
        return(PRED, cost)
