#!/usr/bin/env python3

import numpy as np
class Exponential:
    def __init__(self, data=None, lambtha=1.):
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            
            self.lambtha = sum(data)/len(data)
    
"""    def pmf(self, k):
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return (0)
        else:
            k_fact = 1
            a = 2.7182818285 ** (-1 * self.lambtha)
            b = self.lambtha ** k
            k_temp = k
            while k_temp > 0:
                k_fact = k_temp * k_fact
                k_temp = k_temp - 1 
            return (a * b / k_fact)

    def cdf(self, k):
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return (0)
        else:
            a = 2.7182818285 ** (-1 * self.lambtha)
            suma = 0
            for i in range(0, k + 1, 1):
                num = self.lambtha ** i
                den = 1
                for j in range (1, i + 1, 1):
                    den = den * j
                d = num / den
                suma = suma + d
            return (a * suma)"""
