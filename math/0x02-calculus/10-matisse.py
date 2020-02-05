#!/usr/bin/env python3
""" function poly derivative
"""


def poly_derivative(poly):
    """function poly derivative """
    a = len(poly)
    b = []
    if type(poly) is not list:
        return None
    elif a is 0:
        return None
    elif not all(isinstance(m, (int, float)) for m in poly):
        return None
    else:
        for i in range(1, a):
            b.append(i * poly[i])
        if len(set(b)) == 1:
            return[0]
        return (b)
