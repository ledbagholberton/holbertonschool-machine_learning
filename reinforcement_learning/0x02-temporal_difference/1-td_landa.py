import numpy as np


def learn(state, nxState, reward, weights, gamma, alpha, z):
    delta = reward + gamma * weights[nxState] - weights[state]
    delta = delta * alpha
    weights = weights + delta * z
    return weights


def td_lambtha(env, V, policy, lambtha=0.8, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """ td_lambtha
    env is the openAI environment instance
    V is a numpy.ndarray of shape (s,) containing the value estimate
    policy is a function that takes in a state and returns the next
    action to take
    lambtha is the eligibility trace factor
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    Returns: V, the updated value estimate
    """
    weights = np.zeros(V.shape[0])
    z = np.zeros(V.shape[0])
    for _ in range(episodes):
        s = env.reset()
        action = policy(s)
        for _ in range(max_steps):
            nxState, reward, done, _ = env.step(action)
            dev = 1
            z = z * gamma * lambtha
            z[s] = z[s] + dev
            weights = learn(s, nxState, reward, weights, gamma, alpha, z)
            if done:
                break
            state = nxState
            action = policy(state)
    return V
