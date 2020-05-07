#!/usr/bin/env python3
"""
Write a function  that calculates the definiteness of a matrix

matrix is a numpy.ndarray of shape (n, n) whose definiteness
should be calculated
If matrix is not a numpy.ndarray, raise a TypeError with the message
matrix must be a numpy.ndarray
If matrix is not a valid matrix, return None
Return: the string Positive definite, Positive semi-definite, Negative
semi-definite, Negative definite, or Indefinite if the matrix is positive
definite, positive semi-definite, negative semi-definite, negative definite
of indefinite, respectively
If matrix does not fit any of the above categories, return None
"""
import numpy as np


def definiteness(matrix):
    """Function definiteness"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) is not 2 or (matrix.shape[0] is not matrix.shape[1]):
        return None
    w, v = np.linalg.eig(matrix)
    if all([m > 0 for m in w]):
        return "Positive definite"
    elif all([m >= 0 for m in w]):
        return "Positive semi-definite"
    elif all([m < 0 for m in w]):
        return "Negative definite"
    elif all([m <= 0 for m in w]):
        return "Negative semi-definite"
    else:
        return "Indefinite"
