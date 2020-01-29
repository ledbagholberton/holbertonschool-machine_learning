#!/usr/bin/env python3
""" Function which concatenate matrix from scratch
"""


import numpy as np


def cat_matrices2D(mat1, mat2, axis=0):
    """Function cat_matrices2D"""
    a = np.array(mat1)
    b = np.array(mat2)
    c = np.concatenate((a, b), axis)
    return(c)
