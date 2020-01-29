#!/usr/bin/env python3
""" Function which take two matrix and  return
sum, difference, multiplication and division
"""


import numpy as np


def np_elementwise(mat1, mat2):
    """ Function with basic operation in Numpy"""
    suma = mat1 + mat2
    resta = mat1 - mat2
    multi = mat1 * mat2
    div = mat1 / mat2
    return(suma, resta, multi, div)
