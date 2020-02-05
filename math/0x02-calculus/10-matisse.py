#!/usr/bin/env python3
""" function poly derivative
"""


def poly_derivative(poly):
    """function poly derivative """
    a = len(poly)
    b = []
    try:
        if type(poly) is not list:
            return None
        elif a is 0:
            return None
        elif type(sum(poly)) is not (int or float):
            return None
        else:
            for i in range(1, a):
                b.append(i * poly[i])
            if len(set(b)) == 1:
                return[0]
            return (b)
    except TypeError:
        return(None)
