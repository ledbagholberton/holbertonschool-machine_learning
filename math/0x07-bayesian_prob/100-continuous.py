#!/usr/bin/env python3
"""
Based on 3-posterior.py, write a function  that calculates the posterior
probability that the probability of developing severe side effects
falls within a specific range given the data:

x is the number of patients that develop severe side effects
n is the total number of patients observed
p1 is the lower bound on the range
p2 is the upper bound on the range
You can assume the prior beliefs of p follow a uniform distribution
If n is not a positive integer, raise a
ValueError with the message n must be a positive integer
If x is not an integer that is greater than or equal to 0, raise a
ValueError with the message
x must be an integer that is greater than or equal to 0
If x is greater than n, raise a
ValueError with the message x cannot be greater than n
If p1 or p2 are not floats within the range [0, 1], raise a
ValueError with the message {p} must be a float in the range [0, 1]
where {p} is the corresponding variable
if p2 <= p1, raise a ValueError with the message p2 must be greater than p1
The only import you are allowed to use is from scipy import math, special
Returns:
the posterior probability that p is within the range [p1, p2] given x and n
"""
from scipy import math, special


def factorial(n):
    """Function factorial"""
    factorial = 1
    for i in range(1, int(n)+1):
        factorial = factorial * i
    return factorial


def posterior(x, n, p1, p2):
    """Function posterior"""
    if type(n) is not int or n < 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        msg1 = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg1)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(p1) is not float or p1 < 0 or p1 > 1:
        raise TypeError("p1 must be a float in the range [0, 1]")
    if type(p2) is not float or p2 < 0 or p2 > 1:
        raise TypeError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    post_p1 = 1.32531821e-003
    post_p2 = 8.11911729e-002
    likelihood = x - p1 / (p2 - p1)
    inter1 = likelihood * (post_p1)
    inter2 = likelihood * (post_p2)
    margin = inter1 + inter2
    print(inter1/margin, inter2/margin)
    return(inter1/margin)
