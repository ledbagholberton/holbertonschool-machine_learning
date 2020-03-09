#!/usr/bin/env python3
""" FUnction F1 Score
calculates the F1 score of a confusion matrix:

confusion is a confusion numpy.ndarray of shape (classes, classes)
where row indices represent the correct labels and column indices
represent the predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing the F1 score
of each class
You may use sensitivity = __import__('1-sensitivity').sensitivity and
precision = __import__('2-precision').precision
"""
import numpy as np


def f1_score(confusion):
    """Function F1 Score"""
    sum_x = np.sum(confusion, axis=0)
    sum_y = np.sum(confusion, axis=1)
    sum_all = np.sum(confusion)
    TP = np.diagonal(confusion)
    recall = TP / (sum_x)
    precision = TP / (sum_y)
    f_uno = 2 * (precision * recall) / (precision + recall)
    return(f_uno)
