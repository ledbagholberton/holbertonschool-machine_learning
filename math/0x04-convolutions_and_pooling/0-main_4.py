#!/usr/bin/env python3

import numpy as np
convolve_channels = __import__('4-convolve_channels').convolve_channels

np.random.seed(4)
m = np.random.randint(100, 200)
h, w = np.random.randint(20, 50, 2).tolist()
c = np.random.randint(2, 5)
fh, fw = (np.random.randint(2, 5, 2)).tolist()
sh, sw = (np.random.randint(2, 4, 2)).tolist()

images = np.random.randint(0, 256, (m, h, w, c))
kernel = np.random.randint(0, 10, (fh, fw, c))
print("image shape", images.shape)
print("kernel shape", kernel.shape)
print("stride", sh, sw)
conv_ims = convolve_channels(images, kernel, stride=(sh, sw))
np.set_printoptions(threshold=np.inf)
print(conv_ims)
print(conv_ims.shape)
