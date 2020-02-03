#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]
fig = plt.figure()
data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)
ax = fig.add_subplot(111, projection='3d')
for row in pca_data:
    mark ='o'
    if row[0] < -2:
        color = 'blue'
    elif row[0] >= -2 and row[0] < 2 :
        color = 'red'
    else:
        color = 'yellow'
    ax.scatter(row[0], row[1], row[2], cmap = 'plasma', marker=mark)
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.title('PCA of Iris Dataset')
plt.show()
