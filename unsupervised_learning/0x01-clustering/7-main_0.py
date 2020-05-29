#!/usr/bin/env python3

import numpy as np
maximization = __import__('7-maximization').maximization

if __name__ == '__main__':
    np.random.seed(1)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    g = np.random.randn(4, 12500)
    g = g / np.sum(g, axis=0, keepdims=True)
    pi, m, S = maximization(X, g)
    print(pi)
    print(m)
    print(S)
