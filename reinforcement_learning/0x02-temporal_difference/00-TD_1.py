#!/usr/bin/env python3
from random import seed
from random import random
from matplotlib import pyplot
"""
Generating a Randon Walk
"""
seed(4)
RandomWalk = list()
RandomWalk.append(-1 if random() < 0.5 else 1)
pyplot.plot(RandomWalk)
for i in range(1, 1000):
    Zn = -1 if random() < 0.5 else 1
    Xn = RandomWalk[i-1] + Zn
    RandomWalk.append(Xn)
pyplot.plot(RandomWalk)
pyplot.show()