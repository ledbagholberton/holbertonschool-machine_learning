#!/usr/bin/env python3
""" Normal class
"""


class Normal:
    """ Class Normal """
    def __init__(self, data=None, mean=0., stddev=1.):
        """ Settings for class Normal"""
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data)/len(data)
            suma = 0
            for i in range(0, len(data), 1):
                a = (abs(data[i] - self.mean))**2
                suma = suma + a
            self.stddev = (suma / len(data))**0.5

    def z_score(self, x):
        """Method to calculte z_score for Normal dist"""
        a = (x - self.mean) / self.stddev
        return (a)

    def x_value(self, z):
        """Method to calculate x_value for Normal Dist"""
        a = self.stddev * z + self.mean
        return (a)

    def pdf(self, x):
        """Method to calculate the PDF for a normal distribution """
        pi = 3.1415926536
        e = 2.7182818285
        k1 = 1 / ((2 * pi * (self.stddev**2))**0.5)
        k2 = (-1 * ((x - self.mean)**2))/(2 * (self.stddev**2))
        return (k1 * e ** k2)

    def cdf(self, x):
        """Method to calculate the CDF for a normal distribution """
        pi = 3.1415926536
        sq_dos = 2**0.5
        k1 = (x - self.mean) / (sq_dos * self.stddev)
        erf_k1 = ((4/pi)**0.5)*(k1-(k1**3)/3+(k1**5)/10-k1**7/42+(k1**9)/216)
        return((1+erf_k1)/2)
