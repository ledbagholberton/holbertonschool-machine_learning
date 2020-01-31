#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

X, Y = np.meshgrid(x, y)
Z = np.random.rand(2000) + 40 - np.sqrt(np.square(X) + np.square(Y))
plt.xlabel('x coordinate(m)')
plt.ylabel('y coordinate(m)')
plt.title('Mountain Elevation')

cp = plt.contour(X, Y, Z, colors='black', linestyles='dashed', linewidths=1)
plt.clabel(cp, inline=1, fontsize=10)
cp = plt.contourf(X, Y, Z, )
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
