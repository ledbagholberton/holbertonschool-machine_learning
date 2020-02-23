#!/usr/bin/env python3
""" Function One-hot decode"""
import numpy as np


def one_hot_decode(one_hot):
    """Function one-hot decode"""
    if type(one_hot) is not np.ndarray:
        return None
    if len(one_hot.shape) is not 2:
        return None
    if not(np.amax(one_hot) == 1 and np.amin(one_hot) == 0):
        return None
    row = one_hot.shape[0]
    col = one_hot.shape[1]
    arr = np.ndarray(shape=(col), dtype=int)
    for i in range(row):
        for j in range(col):
            if (one_hot[i][j] == 1):
                arr[j] = i
    return (arr)
