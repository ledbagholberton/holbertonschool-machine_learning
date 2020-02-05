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
        print("poly C is not list")
        return None
    elif type(C) is not (int or float):
        print("C is not int or float")
        return None
    elif a is 0:
        print("len poly is 0")
        return None
    elif type(sum(poly)) is not (int or float):
        print("poly has an element not int or float")
        return None
    else:
        for i in range(0, a):
            b.append(poly[i] / (i + 1))
        return (b)
