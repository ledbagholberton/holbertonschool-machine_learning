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


def viterbi(Observation, Emission, Transition, Initial):
    """calculates the most likely sequence of hidden states
    for a hidden markov model"""
    try:
        T = Observation.shape[0]
        N = Transition.shape[0]
        omega = np.zeros((T, N))
        omega[0, :] = np.log(Initial.T * Emission[:, Observation[0]])
        prev = np.zeros((T - 1, N))
        for t in range(1, T):
            for j in range(N):
                # Same as Forward Probability
                a1 = omega[t-1]
                a2 = np.log(Transition[:, j])
                a3 = np.log(Emission[j, Observation[t]])
                probability = (omega[t - 1] + np.log(Transition[:, j]) +
                               np.log(Emission[j, Observation[t]]))
                # Most probable state given previous state at time t (1)
                prev[t - 1, j] = np.argmax(probability)
                # This is the probability of the most probable state (2)
                omega[t, j] = np.max(probability)
        # Path Array
        S = np.zeros(T)
        # Find the most probable last hidden state
        last_state = np.argmax(omega[T - 1, :])
        S[0] = last_state
        backtrack_index = 1
        for i in range(T - 2, -1, -1):
            S[backtrack_index] = prev[i, int(last_state)]
            last_state = prev[i, int(last_state)]
            backtrack_index += 1
        # Flip the path array since we were backtracking
        S = np.flip(S, axis=0)
        # Convert numeric values to actual hidden states
        result = []
        for s in S:
            result.append(int(s))
        P = np.max(np.exp(omega[-1:, :]))
        return (P, result)
    except Exception:
        return None, None
