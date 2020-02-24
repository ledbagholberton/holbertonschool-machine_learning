#!/usr/bin/env python3
""" Function One-hot encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """Function one-hot encode"""
    if type(classes) is not int:
        return (None)
    if classes < 1:
        return (None)
    if type(Y) is not np.ndarray:
        return (None)
    if len(Y) == 0:
        return (None)
    if np.amax(Y) >= classes:
        return (None)
    matrix = np.zeros((classes, len(Y)))
    for i in range(len(Y)):
        a = Y[i]
        matrix[a][i] = 1
    return (matrix)
