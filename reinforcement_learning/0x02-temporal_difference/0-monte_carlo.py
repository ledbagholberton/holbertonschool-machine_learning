#!/usr/bin/env python3
"""
MonteCarlo Function
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=10000, max_steps=200,
                alpha=0.1, gamma=0.99):
    """
    MonteCarlo Function
    """
    for episode in range(episodes):
        s = env.reset()
        states = [s]
        R = []
        for step in range(max_steps):
            action = policy(s)
            s, reward, done, _ = env.step(action)
            if done:
                R.append(100)
                break
            R.append(reward)
            states.append(s)
        R = np.array(R)
        print(R)
        T = R.shape[0]
        weights = np.logspace(0, T-1, T, base=gamma)
        G = np.zeros((T,))
        for t in range(T):
            G[t] = np.sum(R[t:] * weights[:T-t])
        for t, s in enumerate(states[:-1]):
            V[s] += alpha * (G[t] - V[s])
    return V
