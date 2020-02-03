#!/usr/bin/env python3

def poly_derivative(poly):
    a = len(poly)
    b = []
    for i in poly:
        b.append(a * poly[i])
        a =- 1
    return (b)
