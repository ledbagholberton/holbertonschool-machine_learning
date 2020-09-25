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
    df = pd.DataFrame(data=array)
    return df
