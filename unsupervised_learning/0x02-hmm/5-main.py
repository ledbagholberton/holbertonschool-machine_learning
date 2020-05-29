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
backward = __import__('5-backward').backward


if __name__ == '__main__':
    states = np.array(['ruana', 'saco',
                       'blusa', 'camisa', 'corto', 'neglige'])
    hidden_state = np.array(['helado', 'medio-helado', 'normal', 'medio-hot',
                             'hot'])
    S = states.shape[1]
    N = hidden_state.shape[1]
    Initial = np.array((1, N)) / N
    Transition = np.array([[0.6, 0.39, 0.01, 0, 0],
                          [0.2, 0.5, 0.3, 0, 0],
                          [0.01, 0.24, 0.5, 0.24, 0.01],
                          [0, 0, 0.15, 0.7, 0.15],
                          [0, 0, 0.01, 0.39, 0.6]])
    Emission = np.array([[0.9, 0.1, 0, 0, 0, 0],
                         [0.4, 0.5, 0.1, 0, 0, 0],
                         [0, 0.25, 0.5, 0.25, 0, 0],
                         [0, 0, 0.05, 0.7, 0.15, 0.1],
                         [0, 0, 0, 0.2, 0.5, 0.3]])
    np.random.seed(0)
    Observation = np.random.randint(0, N, 100)
    P, B = backward(Observation, Transition, Emission, Initial)
    print(P, path)
