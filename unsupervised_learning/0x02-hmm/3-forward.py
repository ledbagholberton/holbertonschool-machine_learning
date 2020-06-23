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


def forward(Observations, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model
    tabla transicion probability matrix:
                  helado medio-helado normal medio-hot  hot
    helado          0.6     0.39       0.01
    medio-helado    0.2     0.5        0.3
    normal          0.01    0.24       0.5     0.24     0.01
    medio-hot                          0.15    0.7      0.15
    hot                                0.01    0.39     0.6
    tabla Emission
                  helado medio-helado normal medio-hot  hot
    ruana           0.9       0.4
    saco            0.1       0.5      0.25
    blusa                     0.1      0.5      0.05
    camisa                             0.25     0.7      0.2
    corto                                       0.15     0.5
    neglige                                     0.1      0.3
    """
    try:
        T = Observations.shape[0]
        N = Transition.shape[0]
        alpha = np.zeros((N, T))
        test = Emission[:, Observations[0]]
        alpha[:, 0] = Initial.T * Emission[:, Observations[0]]
        for t in range(1, T):
            for n in range(N):
                a1 = alpha[:, t-1] * Transition[:, n]
                alpha[n, t] = np.sum(Transition[:, n] *
                                     alpha[:, t-1] *
                                     Emission[n, Observations[t]])
        P = np.sum(alpha[:, -1:])
        return P, alpha
    except Exception:
        return None, None
