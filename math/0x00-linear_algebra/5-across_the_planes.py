#!/usr/bin/pyhton3
"""Function which add matrix from scratch
"""


import numpy as np


def add_matrices2D(mat1, mat2):
    """Function add_matrices2D"""
    if len(mat1) !=  len(mat2):
        return (None)
    else:
        c = mat1 + mat2
        return (c)
