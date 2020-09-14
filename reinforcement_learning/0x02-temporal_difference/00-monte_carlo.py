#!/usr/bin/env python3
"""
MonteCarlo Function
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Montecarlo
    env is the openAI environment instance
    V is a numpy.ndarray of shape (s,) containing the value estimate
    policy is a function that takes in a state and returns next action to take
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    Returns: V, the updated value estimate
    Assuming s is a number
    """
    states_count = np.zeros((V.shape))
    returns_sum = np.zeros(V.shape)
    for ep in range(episodes):
        env.reset()
        full_rew = 0
        episode = gen_episodes(policy, max_steps, env)
        # here I have a list of couples of (SAR)
        list_rewards = []
        for x in episode:
            list_rewards.append(x[2])
        for i, x in episode:
            state = x[0]
            states_count[x[0]] += 1


def gen_episodes(policy, max_steps, env):
    episode = []
    current_state = env.reset()
    steps = 0
    while steps < max_steps:
        action = policy(current_state)
        new_state, reward, done, info = env.step(action)
        episode.append((current_state, action, reward))
        if done:
            break
        else:
            steps += 1
            current_state = new_state
    return episode
