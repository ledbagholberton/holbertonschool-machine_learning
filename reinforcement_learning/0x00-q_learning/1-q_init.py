#!/usr/bin/env python3
"""
Function q_init
"""
import numpy as np


def q_init(env):
    """
    Write a function  that initializes the Q-table:

    env is the FrozenLakeEnv instance
    Returns: the Q-table as a numpy.ndarray of zeros
    """
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    q_table = np.zeros((state_space_size, action_space_size))
    return q_table
