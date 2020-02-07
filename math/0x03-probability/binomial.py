#!/usr/bin/env python3
""" Binomial class
"""


class Binomial:
    """ Class Binomial """
    def __init__(self, data=None, n=1, p=0.5):
        """ Settings for class Binomial"""
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p < 0 or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            suma = 0
            n_temp = self.n
            for i in range(0, len(data), 1):
                a = data[i] / n_temp
                suma = suma + a
            self.p = suma / len(data)
            suma = 0
            for i in range(0, len(data), 1):
                suma = data[i] + suma
            self.n = int(round(suma / len(data)/self.p, 0))

    def pmf(self, k):
        """Method to calculate the PMF for a Binomial distribution """
        if type(k) is not int:
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        n_fact = 1
        for i in range(1, self.n + 1, 1):
            n_fact = n_fact * i
        k_fact = 1
        for i in range(1, k+1, 1):
            k_fact = k_fact * i
        nk_fact = 1
        for i in range(1, self.n-k+1, 1):
            nk_fact = nk_fact * i
        return (n_fact/(k_fact * nk_fact))*(self.p ** k)*(1-self.p)**(self.n-k)

    def cdf(self, k):
        """Method to calculate the PMF for a Binomial distribution """
        if type(k) is not int:
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        suma = 0
        for i in range(0, k + 1, 1):
            suma = suma + self.pmf(i)
        return (suma)
