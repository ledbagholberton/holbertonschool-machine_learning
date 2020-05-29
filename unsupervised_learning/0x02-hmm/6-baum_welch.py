#!/usr/bin/env python3
"""
Write the function  that :

Observation is a numpy.ndarray of shape (T,) that contains
the index of the observation
T is the number of observations
N is the number of hidden states
M is the number of possible observations
Transition is the initialized transition probabilities, defaulted to None
Emission is the initialized emission probabilities, defaulted to None
Initial is the initiallized starting probabilities, defaulted to None
If Transition, Emission, or Initial is None, initialize the probabilities as
being a uniform distribution
Returns: the converged Transition, Emission, or None, None on failure
"""
import numpy as np
backward = __import__('5-backward').backward
forward = __import__('3-forward').forward



def baum_welch(Observations, N, M,
               Transition=None, Emission=None, Initial=None):
    """performs the Baum-Welch algorithm for a hidden markov model"""
    T = len(Observations)
    a = Transition
    b = Emission
    while CONDITION_CONVERGENCIA:
        old_a = a.copy()
        old_b = b.copy()
        alpha = forward(Observations, a, b, Initial)
        beta = backward(Observations, a, b, Initial)

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = (np.dot(np.dot(alpha[t, :].T, a) *
                                  b[:, V[t + 1]].T, beta[t + 1, :]))
            for i in range(M):
                numerator = (alpha[t, i] * a[i, :] *
                             b[:, V[t + 1]].T * beta[t + 1, :].T)
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))
        CONDITION_CONVERGENCIA = np.equal(old_a, a) && np.equal(old_b, b)

    return a, b
