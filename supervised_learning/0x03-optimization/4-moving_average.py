#!/usr/bin/env python3
""" Function Moving Average
data is the list of data to calculate the moving average of
beta is the weight used for the moving average
Your moving average calculation should use bias correction
Returns: a list containing the moving averages of data
"""

import numpy as np


def moving_average(data, beta):
    """Function Moving Average"""
    mov_av = []
    a = 0
    for i in range(0, len(data)):
        a = beta * a + (1 - beta) * data[i]
        bias_correction = 1 - (beta ** (i + 1))
        mov_av.append(a / bias_correction)
    return mov_av
