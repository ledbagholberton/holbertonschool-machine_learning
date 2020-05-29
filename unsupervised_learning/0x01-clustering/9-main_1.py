#!/usr/bin/env python3

import numpy as np
BIC = __import__('9-BIC').BIC

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
    best_k, best_result, l, b = BIC(X, kmax=5)
    print(best_k)
    print(best_result)
    print(l)
    print(b)
