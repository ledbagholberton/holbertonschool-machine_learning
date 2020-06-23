"""
Write a function def discriminator(): that creates a discriminator network for MNIST digits:

X is a tf.tensor containing the input to the discriminator network
The network should have two layers:
the first layer should have 128 nodes and use relu activation with name layer_1
the second layer should have 1 node and use a sigmoid activation with name layer_2
All variables in the network should have the scope discriminator with reuse=tf.AUTO_REUSE
Returns Y, a tf.tensor containing the classification made by the discriminator
"""
