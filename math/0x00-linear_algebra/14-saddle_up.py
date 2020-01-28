#!/usr/bin/env python3
""" Function which multiplies two matrix
"""

import numpy as np
def np_matmul(mat1, mat2):
    """Function np_matmul using Numpy"""
    if mat1.shape[1] != mat2.shape[0]:
        return(None)
    else:
        c = mat1.dot(mat2)
        return(c)
    
