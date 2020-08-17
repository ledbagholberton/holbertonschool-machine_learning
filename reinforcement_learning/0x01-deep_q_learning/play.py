#!/usr/bin/env python3
"""
Deep Q Learning from Keras
"""
from __future__ import division

from PIL import Image
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        # resize and convert to grayscale
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        # saves storage in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`.
        # In this case, however,
        # we would need to store a `float32` array instead, which is 4x more
        # memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)
# Get the environment and extract the number of actions.


name = "Breakout-v0"
env = gym.make("Breakout-v0")
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
print(input_shape)
model = Sequential()
model.add(Permute((2, 3, 1), input_shape=input_shape))
model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras
# optimizer and even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a
# random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M
# steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually
# sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing.
# Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the
# agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=.05,
                              nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an
# on-going research topic.
# If you want, you can experiment with the parameters or use a different
# policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
               memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99,
               target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])
weights_filename = 'dqn_{}_weights.h5f'.format(name)
checkpoint_weights_filename = 'dqn_' + name + '_weights_{step}.h5f'
log_filename = 'dqn_{}_log.json'.format(name)
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename,
                                     interval=250000)]
callbacks += [FileLogger(log_filename, interval=100)]
dqn.load_weights('dqn_{}_weights_250000.h5f').format(name)
dqn.test(env, nb_episodes=10, visualize=True)
