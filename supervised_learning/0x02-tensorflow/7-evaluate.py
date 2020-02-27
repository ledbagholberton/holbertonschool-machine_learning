#!/usr/bin/env python3
"""Function Evaluate
This function that builds, trains, and saves a neural network classifier
"""


import tensorflow as tf


def evaluate(X, Y, save_path):
    """Function that evaluates the output of a neural network
    X is a numpy.ndarray containing the input data to evaluate
    Y is a numpy.ndarray containing the one-hot labels for X
    save_path is the location to load the model from
    Returns: the network’s prediction, accuracy, and loss, respectively
    """
    with tf.Session() as session:
        session.run(init)
        counter = 0
        for i in range(iterations):
            acc_t = session.run(accuracy, feed_dict={x: X_train, y: Y_train})
            loss_t = session.run(loss, feed_dict={x: X_train, y: Y_train})
            acc_v = session.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            loss_v = session.run(loss, feed_dict={x: X_valid, y: Y_valid})
            counter = counter + 1
            if (i == 0) or (counter == 100) or (i == iterations - 1):
                print("After {} iterations:".format(i))
                print("Training Cost: {}".format(loss_t))
                print("Training Accuracy: {}".format(acc_t))
                print("Validation Cost: {}".format(loss_v))
                print("Validation Accuracy: {}".format(acc_v))
                counter = 0
            session.run(train_op, feed_dict={x: X_train, y: Y_train})
        save_path = saver.save(session, save_path)
    return(save_path)
