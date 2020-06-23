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
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    N = Transition.shape[0]
    M = Emission.shape[1]
    # Trans_conv, Emm_conv = baum_welch(Observations, N, M, Transition,
    #                                  Emission, Initial.reshape((-1, 1)))
    Trans_conv, Emm_conv = baum_welch(Observations, N, M)
    print(Trans_conv)
    print(Emm_conv)
