#!/usr/bin/env python3

def summation_i_squared(n):
    if n is not 0:
        return (n*n + summation_i_squared(n - 1))
    else:
        return (0)
