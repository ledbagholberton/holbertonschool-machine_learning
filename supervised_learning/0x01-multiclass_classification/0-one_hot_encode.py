#!/usr/bin/env python3
""" Function One-hot encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """Function one-hot encode"""
    matrix = np.zeros((classes, len(Y)))
    if len(Y) is 0 or classes is 0:
        return (None)
    for i in range(len(Y)):
        a = Y[i]
        if a
        matrix[a][i] = 1
    return (matrix)
