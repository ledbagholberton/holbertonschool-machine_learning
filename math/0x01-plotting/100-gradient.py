#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
mio = plt.scatter(x, y, c=z, marker='o')
plt.xlabel('x coordinate(m)')
plt.ylabel('y coordinate(m)')
plt.title('Mountain Elevation')

plt.colorbar(mio, label='elevation (m)', cmap="viridis")
plt.show()
