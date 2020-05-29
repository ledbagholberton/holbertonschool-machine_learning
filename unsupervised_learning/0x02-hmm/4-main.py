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
Transition[i, j] is probability of transitioning from the hidden state i to j
Initial a numpy.ndarray of shape (N, 1) containing
the probability of starting in a particular hidden state
Returns: path, P, or None, None on failure
path is the a list of length T containing
the most likely sequence of hidden states
P is the probability of obtaining the path sequence
"""
import numpy as np
viterbi = __import__('4-viterbi').viterbi


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
    P, path = viterbi(Observation, Emission, Transition, Initial)
    print(P, path)
