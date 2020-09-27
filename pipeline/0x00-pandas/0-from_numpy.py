#!/usr/bin/env python3
"""
function that creates a pd.DataFrame from a np.ndarray
array is the np.ndarray from which you should create the pd.DataFrame
The columns of the pd.DataFrame should be labeled in
alphabetical order and capitalized. There will not be more than 26 columns.
Returns: the newly created pd.DataFrame
"""
import pandas as pd
import numpy as np


def from_numpy(array):
    """
    Creates Dataframe from an np array
    """
    list_alpha = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    len_arr = array.shape[1]
    labels = list_alpha[0: len_arr]
    df = pd.DataFrame(data=array, columns=labels)
    return df
