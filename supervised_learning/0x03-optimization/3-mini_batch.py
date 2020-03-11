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

import tensorflow as tf
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
        m = X_train.shape[0]
        for epoch in range(epochs):
            train_cost = session.run(loss, feed_dict=setTrain)
            train_accuracy = session.run(accuracy, feed_dict=setTrain)
            valid_cost = session.run(loss, feed_dict=setValid)
            valid_accuracy = session.run(accuracy, feed_dict=setValid)
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            X, Y = shuffle_data(X_train, Y_train)
            resto = m % batch_size
            for st in range(0, int(m/batch_size) + 1):
                X_batch = X[st * batch_size: (st+1) * batch_size - 1]
                Y_batch = Y[st * batch_size: (st+1) * batch_size - 1]
                if resto != 0 and st > int(m/batch_size):
                    X_batch = X[st * batch_size:]
                    Y_batch = Y[st * batch_size:]
                setBatch = {x: X_batch, y: Y_batch}
                session.run(train_op, feed_dict=setBatch)
                if st % 100 == 0 and st != 0:
                    step_accuracy = session.run(accuracy, feed_dict=setBatch)
                    step_cost = session.run(loss, feed_dict=setBatch)
                    print("\tStep {}:".format(st))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))
        print("After {} epochs:".format(epoch + 1))
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))
        save_path = saver.save(session, save_path)
    return(save_path)
