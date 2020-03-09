#!/usr/bin/env python3
""" FUnction Specificity
calculates the specificity for each class in a confusion matrix:
confusion is a confusion numpy.ndarray of shape (classes, classes)
where row indices represent the correct labels and column indices
represent the predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing the specificity
of each class
"""
import numpy as np


def specificity(confusion):
    """Function specificity"""
    sum_x = np.sum(confusion, axis=0)
    sum_y = np.sum(confusion, axis=1)
    sum_all = np.sum(confusion)

    TN = sum_all - sum_x - sum_y + np.diagonal(confusion)
    TN_FP = sum_all - sum_y
    spec = TN / TN_FP
    return(spec)
