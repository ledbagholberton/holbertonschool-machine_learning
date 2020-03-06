#!/usr/bin/env python3
""" Function Mini-batch
X_train is a numpy.ndarray of shape (m, 784) with the training data
m is the number of data points
784 is the number of input features
Y_train is a one-hot numpy.ndarray of shape (m, 10) with the training labels
10 is the number of classes the model should classify
X_valid is a numpy.ndarray of shape (m, 784) with the validation data
Y_valid is a one-hot numpy.ndarray of shape (m, 10) with the validation labels
batch_size is the number of data points in a batch
epochs is the number of times the training should pass through whole dataset
load_path is the path from which to load the model
save_path is the path to where the model should be saved
Returns: the path where the model was saved
"""

import numpy as np
import tensorflow as tf
calculate_accuracy = __import__('3-all').calculate_accuracy
calculate_loss = __import__('3-all').calculate_loss
create_placeholders = __import__('3-all').create_placeholders
create_train_op = __import__('3-all').create_train_op
forward_prop = __import__('3-all').forward_prop
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Function Mini-batch"""
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(load_path+'.meta')
        saver.restore(session, load_path)
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
        for j in range(epochs + 1):
            print("After {} epochs:".format(j))
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
                if (counter == 100):
                    print("\tStep {}:".format(step_number))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))
                    counter = 0
                counter = counter + 1
                session.run(train_op, feed_dict=setBatch)
            resto = m % 32
            step_number += 1
            X_resto = np.zeros((resto, X_train.shape[1]))
            Y_resto = np.zeros((resto, Y_train.shape[1]))
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
