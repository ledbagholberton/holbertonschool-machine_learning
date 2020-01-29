#!/usr/bin/python3
"""FUnction which return the matrix shape from scratch
"""


def matrix_shape(matrix):
    """Function matrix_shape"""
    a = len(matrix)
    b = len(matrix[0])
    d = len(matrix[0][0])
    c = [a, b, d]
    return(c)
