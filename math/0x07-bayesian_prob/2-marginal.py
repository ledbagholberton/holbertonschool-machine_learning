#!/usr/bin/env python3
"""
Based on 1-intersection.py, write a function that calculates
the marginal probability of obtaining the data:
x is the number of patients that develop severe side effects
n is the total number of patients observed
P is a 1D numpy.ndarray containing the various hypothetical probabilities
of patients developing severe side effects
Pr is a 1D numpy.ndarray containing the prior beliefs about P
If n is not a positive integer, raise a
ValueError with the message n must be a positive integer
If x is not an integer that is greater than or equal to 0, raise a
ValueError with the message
x must be an integer that is greater than or equal to 0
If x is greater than n, raise a
ValueError with the message x cannot be greater than n
If P is not a 1D numpy.ndarray, raise a
TypeError with the message P must be a 1D numpy.ndarray
If Pr is not a numpy.ndarray with the same shape as P, raise a
TypeError with the message Pr must be a numpy.ndarray with the same shape as P
If any value in P or Pr is not in the range [0, 1], raise a
ValueError with the message
All values in {P} must be in the range [0, 1]
where {P} is the incorrect variable
If Pr does not sum to 1, raise a
ValueError with the message Pr must sum to 1
All exceptions should be raised in the above order
Returns: the marginal probability of obtaining x and n
"""
import numpy as np


def factorial(n):
    """Function factorial"""
    factorial = 1
    for i in range(1, int(n)+1):
        factorial = factorial * i
    return factorial


def marginal(x, n, P, Pr):
    """Function marginal"""
    if type(n) is not int or n < 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        msg1 = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg1)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) and len(P.shape) is not 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.amax(P) > 1 or np.amin(P) < 0:
        raise ValueError("All values in P must be in the range [0, 1]")
    if not isinstance(Pr, np.ndarray):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.amax(P) > 1 or np.amin(P) < 0:
        raise ValueError("All values in P must be in the range [0, 1]")
    sum_p = np.sum(Pr)
    a = np.isclose([sum_p], [1], atol=0)
    if np.all(a) is False:
        raise ValueError("Pr must sum to 1")
    comb = factorial(n)/(factorial(x)*factorial(n-x))
    likelihood = comb * np.power(P, x) * np.power((1-P), n-x)
    inter = likelihood * Pr
    return(np.sum(inter))