#!/usr/bin/python3
"""Function which add arrays
"""


def add_arrays(arr1, arr2):
    """Function add_arrays"""
    if len(arr1) is not len(arr2):
        return None
    else:
        for i in range(len(arr1)):
            a[i] = arr1[i] + arr2[i]
        return (list(a))
