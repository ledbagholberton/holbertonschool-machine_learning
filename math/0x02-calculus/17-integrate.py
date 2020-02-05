#!/usr/bin/env python3
""" function poly integral
"""


def poly_integral(poly, C=0):
    """function poly integral"""
    a = len(poly)
    b = []
    if C is not 0:
        b = [C]
    if type(poly) is not list:
        return None
    elif type(C) is not (int or float):
        return None
    elif not all(isinstance(m, (int, float)) for m in poly):
        return None
    else:
        for i in range(0, a):
            x = poly[i] / (i + 1)
            if x.is_integer():
                x = int(x)
            b.append(poly[i] / (i + 1))
        return (b)
