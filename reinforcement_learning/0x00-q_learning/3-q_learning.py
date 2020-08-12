#!/usr/bin/env python3
"""
function that performs Q-learning:
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    env is the FrozenLakeEnv instance
    Q is a numpy.ndarray containing the Q-table
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    epsilon is the initial threshold for epsilon greedy
    min_epsilon is the minimum value that epsilon should decay to
    epsilon_decay is the decay rate for updating epsilon between episodes

    When the agent falls in a hole, the reward should be updated to be -1
    Returns: Q, total_rewards
    Q is the updated Q-table
    total_rewards is a list containing the rewards per episode
    """
    total_rewards = []
    for epi in range(episodes):
        state = env.reset()
        done = False
        rewards = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            Q[state, action] = (Q[state, action] * (1 - alpha) +
                                (reward + gamma*np.max(Q[new_state, :]))*alpha)
            state = new_state
            if done is True:
                if reward == 1:
                    break
                else:
                    rewards = -1
            rewards += reward
        total_rewards.append(rewards)
        epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-epsilon_decay*epi)
    return Q, total_rewards
