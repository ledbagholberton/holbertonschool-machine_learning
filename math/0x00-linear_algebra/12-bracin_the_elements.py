#!/usr/bin/env python3
""" Function which take two matrix and  return
sum, difference, multiplication and division
"""


import numpy as np


def np_elementwise(mat1, mat2):
    """ Function with basic operation in Numpy"""
    suma = np.add(mat1, mat2)
    resta = np.add(mat1, -1* mat2)
    multi = np.prod((mat1, mat2), axis=0)
    div = np.divide(mat1, mat2)
    return(suma, resta, multi, div)

    
