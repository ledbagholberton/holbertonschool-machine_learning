#!/usr/bin/env python3
"""FUnction summation i squared
"""


def summation_i_squared(n):
    """ FUnction summation """
    if n is not 0:
        return (n*n + summation_i_squared(n - 1))
    else:
        return (0)
