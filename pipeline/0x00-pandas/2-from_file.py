#!/usr/bin/env python3
"""
Write a function  that loads data from a
file as a pd.DataFrame

filename is the file to load from
delimiter is the column separator
Returns: the loaded pd.DataFrame
"""
import pandas as pd
import numpy as np


def from_file(filename, delimiter):
    """ Function from_file"""
    df = pd.read_csv(filename, sep=delimiter)
    return df
