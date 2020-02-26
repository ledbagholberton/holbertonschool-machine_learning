#!/usr/bin/env python3
""" Neural Network class: defines a neural network class
with pivate instances attributes
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """ Class Neural networks"""
    def __init__(self, nx, nodes):
        """ Settings for class Neural Networks"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def sigmoid(self, Z):
        """Fnction sigmoid"""
        sigm = 1 / (1 + np.exp(-Z))
        return sigm

    def forward_prop(self, X):
        """Function Forward Propagation"""
        A1 = np.matmul(self.__W1, X) + self.__b1
        H1 = self.sigmoid(A1)
        self.__A1 = H1
        A2 = np.matmul(self.__W2, self.__A1) + self.__b2
        H2 = self.sigmoid(A2)
        self.__A2 = H2
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """Function cost"""
        m = Y.shape[1]
        num_lreg = -1 * (Y * np.log(A) + (1 - Y) *
                         np.log(1.0000001 - A))
        cost = np.sum(num_lreg)/m
        return (cost)

    def evaluate(self, X, Y):
        """Function evaluate"""
        self.forward_prop(X)
        PRED = np.where(self.__A2 >= 0.5, 1, 0)
        return (PRED, self.cost(Y, self.__A2))

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Function Gradient Descent"""
        m2 = A1.shape[1]
        dZ2 = A2 - Y
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m2
        dZT2 = dZ2.T
        dW2 = np.matmul(self.__A1, dZT2) / m2
        m1 = Y.shape[1]
        dg1 = (self.__A1 * (1 - self.__A1))
        dZ1 = np.matmul(self.__W2.T, dZ2) * dg1
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m1
        self.__b1 = self.__b1 - alpha * db1
        dZT1 = dZ1.T
        dW1 = np.matmul(X, dZT1) / m1
        self.__W1 = self.__W1 - (alpha * dW1.T)
        self.__b2 = self.__b2 - alpha * db2
        self.__W2 = self.__W2 - (alpha * dW2.T)
        return()

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Function Train - with more args"""
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
        for i in range(1, iterations + 1):
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
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
