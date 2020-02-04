#!/usr/bin/env python3
""" function poly derivative
"""

def poly_derivative(poly):
    """function poly derivative """
    a = len(poly)
    b = []
    try:
        if type(poly) is list and a is not 0 and type(sum(poly)) is int:
            for i in range(1, a):
                b.append(i * poly[i])
            return (b)
        else:
            return(None)    
    except TypeError:
        return(None)