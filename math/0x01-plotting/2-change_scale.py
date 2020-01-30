#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

colors = (1, 0, 1)

plt.scatter(x, y, c=colors, marker='o')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of C-14')
plt.show()