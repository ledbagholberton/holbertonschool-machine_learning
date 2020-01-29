#!/usr/bin/python3
""" Function which concatenates arrays from scratch
"""

def cat_arrays(arr1, arr2):
    """FUnction concatenate arrays w/o Numpy"""
    c = [*arr1, *arr2]
    return c
