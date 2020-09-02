"""
Write a function def sample_Z(m, n): that creates input for the generator:

m is the number of samples that should be generated
n is the number of dimensions of each sample
All samples should be taken from a random uniform distribution within the range [-1, 1]
Returns: Z, a numpy.ndarray of shape (m, n) containing the uniform samples
"""
import numpy as np


def sample_Z(m, n):
    """Creates input for generator"""
    return(np.random.uniform(-1, -1, size=[m,n]))
