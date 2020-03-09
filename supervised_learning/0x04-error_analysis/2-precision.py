#!/usr/bin/env python3
""" FUnction Precision
calculates the precision for each class in a confusion matrix:
confusion is a confusion numpy.ndarray of shape (classes, classes)
where row indices represent the correct labels and column indices
represent the predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing the precision
of each class
"""
import numpy as np


def precision(confusion):
    """Function precision"""
    TP = np.diagonal(confusion)
    TP_FP = np.sum(confusion, axis=0)
    prec = TP / TP_FP
    return(prec)
