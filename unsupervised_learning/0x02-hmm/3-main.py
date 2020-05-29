#!/usr/bin/env python3
"""
Observation is a numpy.ndarray of shape (T,) that contains the index of the
observation
T is the number of observations
Emission is a numpy.ndarray of shape (N, M) containing the emission probability
of a specific observation given a hidden state
Emission[i, j] is the probability of observing j given the hidden state i
N is the number of hidden states
M is the number of all possible observations
Transition is a 2D numpy.ndarray of shape (N, N) containing the transition
probabilities
Transition[i, j] is the probability of transitioning from
the hidden state i to j
Initial a numpy.ndarray of shape (N, 1) containing the probability of starting
in a particular hidden state
Returns: P, F, or None, None on failure
P is the likelihood of the observations given the model
F is a numpy.ndarray of shape (N, T) containing the forward path probabilities
F[i, j] is the probability of being in hidden state i at time j given the
previous observations
"""
import numpy as np
forward = __import__('3-forward').forward


if __name__ == '__main__':
    states = np.array(['ruana', 'saco',
                       'blusa', 'camisa', 'corto', 'neglige'])
    hidden_state = np.array(['helado', 'medio-helado', 'normal', 'medio-hot',
                       'hot'])
    
    N = hidden_state.shape[0]
    Initial = np.ones((N, 1)) / N
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
    M = Emission.shape[1]
    np.random.seed(0)
    # Observation = np.random.randint(1, M, 100)
    Observation = np.array([0, 0, 0, 1, 0, 1, 1, 2, 2, 2, 2, 3, 2, 2, 2, 3, 3,
                            3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 5, 5, 5, 5, 4, 3])
    alpha = forward(Observation, Emission, Transition, Initial)
    print(alpha)
