#!/usr/bin/env python3
""" Exponential class
"""


class Exponential:
    """ Class Exponential """
    def __init__(self, data=None, lambtha=1.):
        """ Settings for class Expontential"""
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data)/len(data))

    def pdf(self, x):
        """Method pmf for Exponential dist"""
        if x < 0:
            return (0)
        else:
            a = self.lambtha * 2.7182818285 ** (-1 * self.lambtha * x)
            return (a)

    def cdf(self, x):
        """Method CDF for Exponential Dist"""
        if x < 0:
            return (0)
        else:
            a = 1 - 2.7182818285 ** (-1 * self.lambtha * x)
            return (a)
