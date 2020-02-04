#!/usr/bin/env python3

def poly_derivative(poly):
    a = len(poly)
    print("la longitud es", a)
    b = []
    for i in range(a - 1):
        print(b)
        print(poly[i])
        b.append((a - 1)* poly[a - 1])
        a = a - 1
    return (b)
