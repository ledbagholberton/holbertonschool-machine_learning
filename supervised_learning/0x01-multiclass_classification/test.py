#!/usr/bin/env python3

import numpy as np
Deep = __import__('2-deep_neural_network').DeepNeuralNetwork
DeepTest = __import__('2-deep_neural_network_test').DeepNeuralNetwork

np.random.seed(3)
nx, m = np.random.randint(100, 200, 2).tolist()
X = np.random.randn(nx, m)
Y = np.random.randint(0, 2, (1, m))
deep = Deep(nx, [3, 1])
deep.train(X, Y, iterations=10, graph=False, verbose=False)
deep.save('0-test.pkl')
del deep
deep = DeepTest.load('0-test.pkl')
np.set_printoptions(threshold=np.inf)
print(deep.L)
for k, v in sorted(deep.cache.items()):
    print(k, v)
for k, v in sorted(deep.weights.items()):
    print(k, v)
    