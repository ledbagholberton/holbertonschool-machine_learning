"""
Z is the tf.placeholder that is the input for the generator
X is the tf.placeholder that is the input for the discriminator
The generator should minimize the negative modified minimax loss
The generator should be trained using Adam optimization
The discriminator should NOT be trained
Returns: loss, train_op
loss is the generator loss
train_op is the training operation for the generator
"""
import tensorflow as tf
import numpy as np
generator = __import__('0-generator').generator


def train_generator(Z):
    """ create loss tensor and training op for generator"""
    D_fake = generator(Z)
    G_loss = -tf.reduce_mean(tf.log(D_fake))
    gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("gen")]
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=gen_vars)
    return (G_loss, G_solver)
