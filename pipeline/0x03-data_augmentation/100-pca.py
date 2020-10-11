#!/usr/bin/env python3
"""
Performs PCA color augmentation as described in the AlexNet paper:

image is a 3D tf.Tensor containing the image to change
alphas a tuple of length 3 containe the amount that each channel should change
Returns the augmented image
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def pca_color(image, alphas):
    """Fucntion PCA Color Augmentation
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(100)
    np.random.seed(100)
    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        alphas = np.random.normal(0, 0.1, 3)
        plt.imshow(pca_color(image, alphas))
        plt.show()"""
    original_shape = image.shape
    renorm_image = np.reshape(image, (image.shape[0]*image.shape[1], 3))
    renorm_image = renorm_image.astype('float32')
    mean = np.mean(renorm_image, axis=0)
    std = np.std(renorm_image, axis=0)
    renorm_image = renorm_image - mean
    renorm_image = renorm_image / std
    cov = np.cov(renorm_image, rowvar=False)
    lambdas, p = np.linalg.eig(cov)
    delta = np.dot(p, (alphas*lambdas).T)
    pca_augmentation_version_renorm_image = renorm_image + delta
    pca_image = pca_augmentation_version_renorm_image*std + mean
    pca_image = np.maximum(np.minimum(pca_image, 255), 0).astype('uint8')
    pca_image = np.reshape(pca_image, original_shape)
    return pca_image
