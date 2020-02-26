#!/usr/bin/env python3
"""Function Train
This function that builds, trains, and saves a neural network classifier
"""


import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """Function that builds, trains, and saves a neural network classifier
    X_train is a numpy.ndarray containing the training input data
    Y_train is a numpy.ndarray containing the training labels
    X_valid is a numpy.ndarray containing the validation input data
    Y_valid is a numpy.ndarray containing the validation labels
    layer_sizes is a list containing number nodes in each layer of the network
    activations is a list containing activation functions for each layer
    alpha is the learning rate
    iterations is the number of iterations to train over
    save_path designates where to save the model
    Returns: the path where the model was saved
    """

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection("y_pred", y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection("accuracy", accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection("loss", loss)
    train_op = create_train_op(loss, alpha)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        counter = 0
        for i in range(iterations):
            acc_t = session.run(accuracy, feed_dict={x:X_train, y:Y_train})
            loss_t = session.run(loss, feed_dict={x:X_train, y:Y_train})
            acc_v = session.run(accuracy, feed_dict={x:X_valid, y:Y_valid})
            loss_v = session.run(loss, feed_dict={x:X_valid, y:Y_valid})
            counter = counter + 1
            if (i == 0) or (counter == 100) or (i == iterations - 1):
                print("After {} iterations:".format(i))
                print("Training Cost: {}".format(loss_t))
                print("Training Accuracy: {}".format(acc_t))
                print("Validation Cost: {}".format(loss_v))
                print("Validation Accuracy: {}".format(acc_v))
                counter = 0
            session.run(train_op, feed_dict={x:X_train, y:Y_train})
        save_path = saver.save(session, save_path)
    return(save_path)
