#!/usr/bin/env python3

import numpy as np
def np_elementwise(mat1, mat2):
    suma = np.add(mat1, mat2)

    resta = np.add(mat1, -1* mat2)


    

    multi = np.prod((mat1, mat2), axis=0)



    div = np.divide(mat1, mat2)

    return(suma, resta, multi, div)

    
