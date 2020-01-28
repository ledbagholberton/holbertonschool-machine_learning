#!/usr/bin/python3
import numpy as np
def cat_arrays(arr1, arr2):
    a = np.array(arr1)
    b = np.array(arr2)
    c = [*a, *b]
    return c
