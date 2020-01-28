#!/usr/bin/env python3
""" Function which multiplies two matrix
"""


import numpy as np


def np_matmul(mat1, mat2):
    """Function np_matmul using Numpy"""
    c = np.matmul(mat1, mat2)
    return(c)
