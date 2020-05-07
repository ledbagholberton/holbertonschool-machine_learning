#!/usr/bin/env python3
"""Function that calculates the determinant of a matrix:

matrix is a list of lists whose determinant should be calculated
If matrix is not a list of lists, raise a TypeError with the
message matrix must be a list of lists
If matrix is not square, raise a ValueError with the message
matrix must be a square matrix
The list [[]] represents a 0x0 matrix
Returns: the determinant of matrix"""


def determinant(matrix):
    """Function determinant"""
    if (type(matrix) != list or len(matrix) == 0 or
       not all([type(m) == list for m in matrix])):
        raise TypeError("matrix must be a list of lists")
    elif len(matrix) is 1:
        if len(matrix[0]) is not 0:
            return(matrix[0][0])
        else:
            return(1)
    elif len(matrix) is not len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    elif len(matrix) is 2:
        return(matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0])
    else:
        total = 0
        for fc in range(len(matrix)):
            As = list(matrix)
            As = As[1:]
            for i in range(len(As)):
                As[i] = As[i][0:fc] + As[i][fc+1:]
            sign = (-1) ** (fc % 2)
            sub_det = determinant(As)
            total += sign * matrix[0][fc] * sub_det
        return total
