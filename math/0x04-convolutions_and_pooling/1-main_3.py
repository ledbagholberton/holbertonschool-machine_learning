#!/usr/bin/env python3

import numpy as np
convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale

np.random.seed(3)
m = np.random.randint(100, 200)
h, w = np.random.randint(20, 50, 2).tolist()
fh, fw = (np.random.randint(2, 5, 2)).tolist()
sh, sw = (np.random.randint(2, 4, 2)).tolist()

images = np.random.randint(0, 256, (m, h, w))
kernel = np.random.randint(0, 10, (fh, fw))
conv_ims = convolve_grayscale(images, kernel, padding='valid', stride=(sh, sw))
print(conv_ims)
print(conv_ims.shape)
