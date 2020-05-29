#!/usr/bin/env python3

import numpy as np
maximization = __import__('7-maximization').maximization

if __name__ == '__main__':
    np.random.seed(1)
    m = np.random.randint(-100, 101, (3, 6))
    S = np.random.randint(-3, 3, (3, 6, 6))
    S = np.matmul(S, S.transpose(0, 2, 1))
    n = np.random.randint(100, 10001, 3)
    a = np.random.multivariate_normal(m[0], S[0], size=n[0])
    b = np.random.multivariate_normal(m[1], S[1], size=n[1])
    c = np.random.multivariate_normal(m[2], S[2], size=n[2])
    X = np.concatenate((a, b, c), axis=0)
    g = np.zeros((3, np.sum(n)))
    g[0, :n[0]] = 1
    g[1, n[0]:-n[2]] = 1
    g[2, -n[2]:] = 1
    g = g + np.random.uniform(0, 0.1, size=(3, np.sum(n)))
    g = g / np.sum(g, axis=0, keepdims=True)
    p = np.random.permutation(np.sum(n))
    X = X[p]
    g = g[:, p]
    pi, m, S = maximization(X, g)
    print(pi)
    print(m)
    print(S)
