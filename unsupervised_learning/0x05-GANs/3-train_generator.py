"""
Write a function def train_generator(Z, X): that creates the loss tensor and training op for the generator:

Z is the tf.placeholder that is the input for the generator
X is the tf.placeholder that is the input for the discriminator
You can use the following imports:
generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator
The generator should minimize the negative modified minimax loss
The generator should be trained using Adam optimization
The discriminator should NOT be trained
Returns: loss, train_op
loss is the generator loss
train_op is the training operation for the generator
"""
