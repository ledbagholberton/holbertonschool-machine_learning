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
    try:
        tol = 1e-10
        T = len(Observations)
        if Transition is None:
            Transition = np.random.uniform(0, 1, size=(N, N))
        if Emission is None:
            Emission = np.random.uniform(0, 1, size=(N, M))
        if Initial is None:
            Initial = np.random.uniform(0, 1, size=(N, 1))
        a = Transition
        b = Emission
        cond = False
        count = 0
        norm_a = 0
        norm_b = 0
        while not cond:
            count = count + 1
            print(count)
            old_norm_a = norm_a
            old_norm_b = norm_b
            old_a = a.copy()
            old_b = b.copy()
            _, alpha = forward(Observations, b, a, Initial)
            _, beta = backward(Observations, b, a, Initial)
            xi = np.zeros((N, N, T - 1))
            for t in range(T - 1):
                denominator = (np.dot(np.dot(alpha[:, t].T, a) *
                                      b[:, Observations[t+1]].T, beta[:, t+1]))
                for i in range(N):
                    numerator = (alpha[i, t] * a[i, :] *
                                 b[:, Observations[t + 1]].T * beta[:, t+1].T)
                    xi[i, :, t] = numerator / denominator
            gamma = np.sum(xi, axis=1)
            a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
            # Add additional T'th element in gamma
            gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2],
                                             axis=0).reshape((-1, 1))))
            K = b.shape[1]
            denominator = np.sum(gamma, axis=1)
            for l in range(K):
                b[:, l] = np.sum(gamma[:, Observations == l], axis=1)
            b = np.divide(b, denominator.reshape((-1, 1)))
            # conditional convergence based on normal over difference of 
            # old and new matrix (Emission & Transition)
            norm_a = np.linalg.norm(np.abs(old_a - a))
            norm_b = np.linalg.norm(np.abs(old_b - b))
            print ("Norm Transition= ", norm_a)
            print ("Norm Emission= ", norm_b)
            print ("Difference in Transition= ", np.abs(old_norm_a - norm_a))
            print ("Difference in Emission= ", np.abs(old_norm_b - norm_b))
            cond = (np.abs(old_norm_a - norm_a) < tol) and (np.abs(old_norm_b == norm_b) < tol)
        return a, b
    except Exception:
        return None, None
