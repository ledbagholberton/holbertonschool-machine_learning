"""
Z is the tf.placeholder that is the input for the generator
X is the tf.placeholder that is the input for the discriminator
You can use the following imports:

The discriminator should minimize the negative minimax loss
The discriminator should be trained using Adam optimization
The generator should NOT be trained
Returns: loss, train_op
loss is the discriminator loss
train_op is the training operation for the discriminator
"""
import tensorflow as tf
import numpy as np 
generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_discriminator(Z, X):
    """Create the loss tensor and training op for the discriminator"""
    D_fake = discriminator(generator(Z))
    D_real = discriminator(X)
    D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
    disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("disc")]
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=disc_vars)
    return (D_loss, D_solver)
