#!/usr/bin/env python3
""" FUnction which concatenate matrix
"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ Function in charge to concatenate """
    c = np.concatenate((mat1, mat2), axis)
    return (c)
