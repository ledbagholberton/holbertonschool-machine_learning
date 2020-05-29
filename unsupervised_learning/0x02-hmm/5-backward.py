#!/usr/bin/env python3
"""
Observation is a numpy.ndarray of shape (T,) that contains
the index of the observation
T is the number of observations
Emission is a numpy.ndarray of shape (N, M) containing
the emission probability of a specific observation given a hidden state
Emission[i, j] is the probability of observing j given the hidden state i
N is the number of hidden states
M is the number of all possible observations
Transition is a 2D numpy.ndarray of shape (N, N) containing
the transition probabilities
Transition[i, j] is
the probability of transitioning from the hidden state i to j
Initial a numpy.ndarray of shape (N, 1) containing
the probability of starting in a particular hidden state
Returns: P, B, or None, None on failure
P is the likelihood of the observations given the model
B is a numpy.ndarray of shape (N, T) containing the backward path probabilities
B[i, j] is the
probability of generating the future observations from hidden state i at time j
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """performs the backward algorithm for a hidden markov model"""

    beta = np.zeros((Observation.shape[0], Transition.shape[0]))

    # setting beta(T) = 1
    beta[Observation.shape[0] - 1] = np.ones((Transition.shape[0]))

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(Observation.shape[0] - 2, -1, -1):
        for j in range(Transition.shape[0]):
            beta[t, j] = ((beta[t + 1] * Emission[  :, Observation[t + 1]])
                          .dot(Transition[j, :]))
    P = 1
    return P, beta
