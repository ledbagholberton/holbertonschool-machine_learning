#!/usr/bin/python3
import numpy as np
def add_arrays(arr1, arr2):
    a = np.array(arr1)
    b = np.array(arr2)
    if len(arr1) is not len (arr2):
        return None
    else:
        for i in range(len(arr1)):
            a[i] = arr1[i] + arr2[i]
        return (list(a))
    

    
