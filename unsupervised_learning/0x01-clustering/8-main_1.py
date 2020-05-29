#!/usr/bin/env python3

import numpy as np
EM = __import__('8-EM').expectation_maximization

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
    np.random.shuffle(X)
    k = 3
    pi, m, S, g, l = EM(X, k, 100, verbose=True)
    print(pi)
    print(m)
    print(S)
    print(g)
    print(l)
