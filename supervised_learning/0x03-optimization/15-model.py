#!/usr/bin/env python3
""" Function Model
builds, trains, and saves a neural network model in tensorflow using 
Adam optimization, mini-batch gradient descent, learning rate decay, 
and batch normalization:
Data_train is a tuple containing the training inputs and
training labels, respectively
Data_valid is a tuple containing the validation inputs and
validation labels, respectively
layers is a list containing the number of nodes in each layer
of the network
activation is a list containing the activation functions used for
each layer of the network
alpha is the learning rate
beta1 is the weight for the first moment of Adam Optimization
beta2 is the weight for the second moment of Adam Optimization
epsilon is a small number used to avoid division by zero
decay_rate is the decay rate for inverse time decay of the learning rate
(the corresponding decay step should be 1)
batch_size is the number of data points that should be in a mini-batch
epochs is the number of times the training should pass through
the whole dataset
save_path is the path where the model should be saved to
Returns: the path where the model was saved
"""
import numpy as np
import tensorflow as tf


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """Function Model"""
    with tf.Session() as session:
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        setTrain = {x: X_train, y: Y_train}
        setValid = {x: X_valid, y: Y_valid}
        train_cost = session.run(loss, feed_dict=setTrain)
        train_accuracy = session.run(accuracy, feed_dict=setTrain)
        valid_cost = session.run(loss, feed_dict=setValid)
        valid_accuracy = session.run(accuracy, feed_dict=setValid)
        m = X_train.shape[0]
        X_shuffled = X_train
        Y_shuffled = Y_train
        X_batch = np.zeros((32, X_train.shape[1]))
        Y_batch = np.zeros((32, Y_train.shape[1]))
        for epochs in range(epochs + 1):
            print("After {} epochs:".format(epochs))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            counter = 0
            for step_number in range(0, int(m/32)):
                for a in range(32):
                    X_batch[a] = X_shuffled[step_number * 32 + a]
                    Y_batch[a] = Y_shuffled[step_number * 32 + a]
                setBatch = {x: X_batch, y: Y_batch}
                step_accuracy = session.run(accuracy, feed_dict=setBatch)
                step_cost = session.run(loss, feed_dict=setBatch)
                counter = counter + 1
                if (counter == 100):
                    print("\tStep {}:".format(step_number + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))
                    counter = 0
                session.run(train_op, feed_dict=setBatch)
            resto = m % 32
            step_number += 1
            X_resto = np.zeros((resto, X_train.shape[1]))
            Y_resto = np.zeros((resto, Y_train.shape[1]))
            print(X_resto)
            print(Y_resto)
            for a in range(resto):
                X_resto[a] = X_shuffled[step_number * 32 + a]
                Y_resto[a] = Y_shuffled[step_number * 32 + a]
            setResto = {x: X_resto, y: Y_resto}
            step_accuracy = session.run(accuracy, feed_dict=setResto)
            step_cost = session.run(loss, feed_dict=setResto)
            session.run(train_op, feed_dict=setResto)
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            train_accuracy = step_accuracy
            train_cost = step_cost
            valid_accuracy = session.run(accuracy, feed_dict=setValid)
            valid_cost = session.run(loss, feed_dict=setValid)
        save_path = saver.save(session, save_path)
    return(save_path)
