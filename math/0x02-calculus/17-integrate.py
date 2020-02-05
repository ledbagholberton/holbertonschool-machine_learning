#!/usr/bin/env python3
""" function poly integral
"""


def poly_integral(poly, C=0):
    """function poly integral"""
    a = len(poly)
    b = [C]
    try:
        if type(poly) is not list:
            return None
        elif type(C) is not (int or float):
            return None
        elif a is 0:
            return None
        elif type(sum(poly)) is not (int or float):
            return None
        else:
            for i in range(0, a):
                b.append(poly[i] / (i + 1))
            if len(set(b)) == 1:
                return[0]
            return (b)
    except TypeError:
        return(None)