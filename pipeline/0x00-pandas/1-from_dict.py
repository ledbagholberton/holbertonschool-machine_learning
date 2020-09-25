#!/usr/bin/env python3
"""
First column should be labeled First and have the values 0.0, 0.5, 1.0, and 1.5
Second column should be labeled Second with values one, two, three, four
Rows should be labeled A, B, C, and D, respectively
The pd.DataFrame should be saved into the variable df
"""
import pandas as pd
import numpy as np


def from_dict(array):
    """
    Creates Dataframe from a dict
    """
    data = {'First': [0.0, 0.5, 1.0,  1.5],
            'Second': ['one', 'two', 'three', 'four']}
    df = pd.DataFrame.from_dict(data, orient='index',
                                columns=['A', 'B', 'C', 'D'])
    return df
