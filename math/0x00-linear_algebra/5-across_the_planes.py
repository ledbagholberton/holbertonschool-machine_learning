#!/usr/bin/pyhton3
import numpy as np
def add_matrices2D(mat1, mat2):
    a = np.array(mat1)
    b = np.array(mat2)
    if a.shape !=  b.shape:
        return (None)
    else:
        c = a + b
        return (c)
