#!/usr/bin/env python3

import gym
import numpy as np
import random as r

Td = __import__('1-td_landa').td_lambtha

UP = 0
RIGHT = 1
LEFT = 2
DOWN = 3
np.random.seed(0)
env = gym.make("CliffWalking-v0")
V = np.zeros((48,))


def policy(state_idx):
    """
    Policy function
    """
    p = np.random.uniform()
    if p < 0.01 or not state_idx:
        random_a = r.randrange(0, 4, 1)
        return (random_a)
    if state_idx // 12 != 0:
        return UP
    if state_idx % 12 == 11:
        return DOWN
    return RIGHT
V = Td(env, V, policy)
print(V)
