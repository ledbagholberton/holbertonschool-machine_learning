#!/usr/bin/env python3
"""
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
baum_welch = __import__('6-baum_welch').baum_welch


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
    M = Emission.shape[1]
    np.random.seed(0)
    Observation = np.random.randint(0, N, 100)
    Trans_conv, Emm_conv = baum_welch(Observation, N, M,
                                      Transition, Emission, Initial)
    print(Trans_conv, Emm_conv)
