#!/usr/bin/env python3
"""
Function epsilon_greedy
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action:
    Q is a numpy.ndarray containing the q-table
    state is the current state
    epsilon is the epsilon to use for the calculation
    You should sample p with numpy.random.uniformn to determine if your
    algorithm should explore or exploit
    If exploring, you should pick the next action with numpy.random.randint
    from all possible actions
    Returns: the next action index
    """
    exploration_rate_threshold = np.random.uniform(0, 1)
    if exploration_rate_threshold > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, 3, None)
    return action
