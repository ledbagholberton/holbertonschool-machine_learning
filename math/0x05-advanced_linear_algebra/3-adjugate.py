#!/usr/bin/env python3
"""
Write a function  that calculates the adjugate matrix of a matrix:

matrix is a list of lists whose adjugate matrix should be calculated
If matrix is not a list of lists, raise a TypeError with the message
atrix must be a list of lists
If matrix is not square or is empty, raise a ValueError with the message
matrix must be a non-empty square matrix
Returns: the adjugate matrix of matrix
"""


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


def minor(matrix):
    """Function minor"""
    if (type(matrix) != list or len(matrix) == 0 or
       not all([type(m) == list for m in matrix])):
        raise TypeError("matrix must be a list of lists")
    elif (len(matrix) is not len(matrix[0])) or len(matrix) is 0:
        raise ValueError("matrix must be a square matrix")
    if (len(matrix) is 1):
        return [[1]]
    if (len(matrix) == 2):
        return([[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]])
    if (len(matrix) > 2):
        total = []
        for i in range(len(matrix)):
            As = []
            for j in range(len(matrix[i])):
                slice_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] +
                                                                matrix[i+1:])]
                As.append(determinant(slice_matrix))
            total.append(As)
        return total


def cofactor(matrix):
    """FUnction cofactor"""
    if (type(matrix) != list or len(matrix) == 0 or
       not all([type(m) == list for m in matrix])):
        raise TypeError("matrix must be a list of lists")
    elif (len(matrix) is not len(matrix[0])) or len(matrix) is 0:
        raise ValueError("matrix must be a square matrix")
    A = minor(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            sign = (-1) ** (i + j)
            A[i][j] = sign * A[i][j]
    return(A)


def adjugate(matrix):
    """ FUnction Adjugate"""
    if (type(matrix) != list or len(matrix) == 0 or
            not all([type(m) == list for m in matrix])):
        raise TypeError("matrix must be a list of lists")
    elif (len(matrix) is not len(matrix[0])) or len(matrix) is 0:
        raise ValueError("matrix must be a square matrix")
    A = cofactor(matrix)
    return (transpose(A))


def transpose(matrix):
    """Function transpose"""
    As = []
    for i in range(len(matrix)):
        total = []
        for j in range(len(matrix[i])):
            total.append(matrix[j][i])
        As.append(total)
    return(As)
