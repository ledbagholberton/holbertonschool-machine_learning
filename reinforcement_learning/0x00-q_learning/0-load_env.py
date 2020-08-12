#!/usr/bin/env python3
"""
Function load_frozen_lake
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    function that loads the pre-made FrozenLakeEnv evnironment from
    OpenAI’s gym

    Parameters
    ----------
    desc : is either None or a list of lists containing a custom description
    of the map to load for the environment
    DESCRIPTION. The default is None.
    map_name : is either None or a string containing the pre-made map to load
        DESCRIPTION. The default is None.
    Note: If both desc and map_name are None, the environment will load a
    randomly generated 8x8 map
    is_slippery : is a boolean to determine if the ice is slippery
        DESCRIPTION. The default is False.
    Returns
    -------
    The environment
    """
    env = gym.make('FrozenLake-v0',
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)
    return env
