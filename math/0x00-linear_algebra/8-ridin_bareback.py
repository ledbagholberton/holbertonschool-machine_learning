#! /usr/bin/env python3
import numpy as np
def mat_mul(mat1, mat2):
    a = np.array(mat1)
    b = np.array(mat2)
    if a.shape[1] != b.shape[0]:
        return(None)
    else:
        c = a.dot(b)
        return(c)
    
