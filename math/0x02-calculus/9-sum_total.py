#!/usr/bin/env python3
"""FUnction summation i squared
"""


def summation_i_squared(n):
    """ FUnction summation """
    if type(n) == int and n > 0:
        return (int((n*(n+1)*(2*n + 1))/6))
    else:
        return (None)
