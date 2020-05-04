#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

NST = __import__('10-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("quilla_1.jpg")

    np.random.seed(0)
    nst = NST(style_image, content_image)
    generated_image, cost = nst.generate_image(iterations=10, step=5, lr=0.002)
    print("Best cost:", cost)
    plt.imshow(generated_image)
    plt.show()
    mpimg.imsave("quilla_generated_2.jpg", generated_image)
