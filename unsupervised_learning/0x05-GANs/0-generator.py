#!/usr/bin/env python3
"""
Write a function that :

Z is a tf.tensor containing the input to the generator network
The network should have two layers:
the first layer should have 128 nodes and use relu activation with name layer_1
the second layer should have 784 nodes and use a sigmoid activation with name layer_2
Returns X, a tf.tensor containing the generated image
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib as plt

def generator(Z):
    """Function generator creates a simple generator network for MNIST digits"""
    
